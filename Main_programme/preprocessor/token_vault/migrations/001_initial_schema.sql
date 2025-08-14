-- Token Vault Initial Schema Migration
-- Creates the core vault_records table with proper indexing

CREATE TABLE IF NOT EXISTS vault_records (
    token_id VARCHAR(32) PRIMARY KEY,
    type VARCHAR(50) NOT NULL,
    scope VARCHAR(20) NOT NULL CHECK (scope IN ('tenant', 'global')),
    tenant_id VARCHAR(100),
    encrypted_data_key BYTEA NOT NULL,
    cipher_value BYTEA NOT NULL,
    nonce BYTEA NOT NULL,
    tag BYTEA NOT NULL,
    salted_hash BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Ensure token_id uniqueness
    UNIQUE(token_id),

    -- Composite constraint for tenant isolation
    CONSTRAINT vault_records_tenant_check
        CHECK (
            (scope = 'global' AND tenant_id IS NULL) OR
            (scope = 'tenant' AND tenant_id IS NOT NULL)
        )
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_vault_records_salted_hash
    ON vault_records(salted_hash);

CREATE INDEX IF NOT EXISTS idx_vault_records_tenant_scope
    ON vault_records(tenant_id, scope);

CREATE INDEX IF NOT EXISTS idx_vault_records_type
    ON vault_records(type);

CREATE INDEX IF NOT EXISTS idx_vault_records_created_at
    ON vault_records(created_at);

-- Comments for documentation
COMMENT ON TABLE vault_records IS 'Secure storage for tokenized PII values with AES-GCM encryption';
COMMENT ON COLUMN vault_records.token_id IS 'Deterministic base32-encoded token identifier';
COMMENT ON COLUMN vault_records.encrypted_data_key IS 'AES data key encrypted with KEK (envelope encryption)';
COMMENT ON COLUMN vault_records.cipher_value IS 'AES-GCM encrypted PII value';
COMMENT ON COLUMN vault_records.salted_hash IS 'SHA-256 hash for fast lookups without plaintext access';
