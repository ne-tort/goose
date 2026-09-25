# Provider Error Retry — архитектура и согласованность слоёв

## Контекст (по итогам исследования)

Ретраи уже существуют в двух местах; новая система не должна с ними конфликтовать:

- **HTTP-слой** (`crates/goose-provider-types/src/retry.rs`): `with_retry` — 3 попытки, экспоненциальный backoff, только транзиентные (`RateLimit`, `ServerError`, `NetworkError`, неперманентный `RequestFailed`). Видит один HTTP-запрос. Настраивается per-provider (bedrock/databricks).
- **Агентный цикл**: legacy `agent.rs` ретраит пустой ответ до 3 раз (`MAX_EMPTY_TURN_RETRIES`); state machine не ретраит ничего — `ExitOnErrorOperation` завершает turn.

## Целевая архитектура

Три слоя, каждый со своей зоной ответственности; верхний слой не вмешивается в работу нижнего до тех пор, пока нижний не исчерпал свои попытки (тогда они и не конфликтуют — верхний получает ошибку уже после того, как HTTP-слой сдался).

```
┌─ Turn-level Provider Retry (НОВЫЙ) ─────────────────────────────────┐
│ Зона: ошибки, дожившие до агентного цикла, и пустые ответы.          │
│ Политика: GOOSE_PROVIDER_ERROR_RETRIES (-1=∞),                       │
│           GOOSE_PROVIDER_RETRY_INTERVAL_SECONDS (фиксированный).     │
│ Живёт: legacy loop (флаг retrying_after_provider_error) +            │
│        state machine (новая операция перед ExitOnError).             │
│ Действие: sleep(интервал, cancellable) → повторить turn.             │
└───────────────────────────────────────────────────────────────────────┘
┌─ Recipe Retry (существует, не меняется) ──────────────────────────────┐
│ Зона: «задача выполнена?» после успешного ответа. success-checks.    │
│ Конфликт исключён: срабатывает только при ends_turn и отсутствии     │
│ trailing_error — т.е. когда провайдер уже ответил.                   │
└───────────────────────────────────────────────────────────────────────┘
┌─ HTTP Retry (существует, не меняется) ────────────────────────────────┐
│ Зона: единичный HTTP-запрос. Короткие блипы (секунды).               │
└───────────────────────────────────────────────────────────────────────┘
```

### Правила согласованности (защита от «скрытия» ошибок)

1. **Единая классификация ретраебельности** — один источник истины: новый модуль `crates/goose/src/agents/provider_retry.rs`. Ретраебельно: пустой ответ, `NetworkError`, `ServerError`, rate-limit-подобные `Other` от провайдера. Терминально: `Authentication`, `CreditsExhausted`, `Refusal`, `ContextLengthExceeded`. Терминальные проходят новый слой нетронутыми — пользователь видит их сразу (как сегодня).
2. **Ничего не скрывается молча**: каждый ретрай turn-уровня пишет `warn!` (попытка N/M, интервал). При исчерпании — сегодняшнее сообщение об ошибке пользователю, без изменений формата.
3. **Не персистить ошибки промежуточных попыток** (legacy: NetworkError/ServerError сегодня и так не персистятся; state machine: `reset_conversation` до kickoff отбрасывает сообщение-ошибку — прецедент `ops_retry.rs`). Финальная ошибка — персистится (state machine: без ретрая → `ExitOnError`; legacy: сегодняшние yield-ы).
4. **Пустой ответ = ретраебельная «ошибка»** в обоих путях (сегодня: ретраится только в legacy, hardcoded 3). Новая политика заменяет `MAX_EMPTY_TURN_RETRIES` дефолтом 3 — поведение по умолчанию не меняется.
5. **Ретраи turn-уровня не съедают turns_taken** (флаг в цепочке инкремента, как `retrying_after_empty_turn`) и **cancellable** (`tokio::select!` на cancel-токен) — очередь/остановка не ломаются.
6. **Порядок операций state machine**: …UnknownTool → **ProviderErrorRetry (новый)** → RetryOperation (recipe) → StopHook → ExitOnError. Recipe retry видит conversation уже без ошибки (reset), но т.к. ProviderErrorRetry возвращает `applied` без yield, recipe не стартует параллельно.
7. **Взаимодействие с очередями UI**: во время ожидания ретрая run активен (`activeRunId` не сброшен) — очередь ждёт; после финальной ошибки run завершается — очередь идёт дальше. Требование п.2 исходной задачи выполняется автоматически.

## Задания субагентам

Общий интерфейс от субагента 1 (остальные зависят от него): модуль `provider_retry.rs`, типы `ProviderRetryPolicy`, `RetryDecision`, функции `load_policy(&Config)`, `classify(MessageErrorKind, &str) -> RetryDecision`.

### Субагент 1 — ядро политики (без зависимостей)
Модуль `crates/goose/src/agents/provider_retry.rs` + регистрация в `agents/mod.rs`. Юнит-тесты (парсинг, классификация, дефолты).

### Субагент 2 — state machine (зависит от 1)
Операция `ops_provider_error_retry.rs` по образцу `ops_retry.rs`; вставка в `agent.rs` (~1722) и `tests/pipeline.rs` (~160); учёт EMPTY_RESPONSE_MESSAGE из `goose-agent/src/inference.rs`.

### Субагент 3 — legacy loop (зависит от 1)
`agent.rs`: чтение политики, флаг `retrying_after_provider_error` (~2667), error-arms (~3253/3263/3388).

### Субагент 4 — тесты state machine (зависит от 2)
`tests/provider_lifecycle.rs` + dummy_api: ретрай→успех, исчерпание, ∞, терминальные.

### Субагент 5 — тесты legacy (зависит от 3)
`crates/goose/tests/agent.rs::empty_turn_tests`-style: counted-mock с NetworkError/ServerError.

### Субагент 6 — документация + self-test (зависит от 1)
env-variables.md, config-files.md, AgentLoopSettings.tsx, goose-self-test.yaml.

### Субагент 7 — верификация (после всех)
fmt, clippy, `cargo test -p goose -p goose-agent`, оба пути, сводный отчёт.

Порядок: 1 → (2‖3) → (4‖5) → 6 → 7.

## Статус: реализовано, верифицировано (2026-09-25)

- Все 7 задач выполнены. Тесты: 12 юнит (policy) + 11 (state machine lifecycle, включая cancel/∞/исчерпание) + 1 юнит (refusal guard) + 8 legacy integration + 10 существующих empty_turn — зелёные; clippy/fmt чистые по затронутым файлам; 20 предсуществующих падений state_machine-тестов не изменились (базлайн master).
- Асимметрия Refusal исправлена: в state machine операция распознаёт "Provider refused request" в тексте ошибки (по образцу PERMANENT_REQUEST_FAILURE_MARKERS) и не ретраит — паритет с legacy.
- Известные мелкие отличия (не влияют на согласованность): SM считает попытки на весь turn (строже), legacy sleep не cancellable при None cancel_token (унаследовано).
