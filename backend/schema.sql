CREATE EXTENSION IF NOT EXISTS citext;

CREATE TABLE users (
    id             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    username       CITEXT      NOT NULL UNIQUE
                   CHECK (char_length(username) BETWEEN 3 AND 32),
    password_hash  TEXT        NOT NULL,
    xp             INTEGER     NOT NULL DEFAULT 0 CHECK (xp >= 0),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE xp_events (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id     BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    amount      INTEGER     NOT NULL CHECK (amount > 0),
    reason      TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX xp_events_user_created_idx ON xp_events (user_id, created_at DESC);
CREATE INDEX users_xp_idx ON users (xp DESC);
