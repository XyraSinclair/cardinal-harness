//! Error types for the provider gateway.

use std::time::Duration;
use thiserror::Error;

/// Source of a rate limit: local (our limiter) or remote (provider 429).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateLimitSource {
    /// Our local rate limiter blocked the request.
    Local,
    /// The provider returned a 429 response.
    Remote,
}

/// Additional context from provider errors for debugging.
#[derive(Debug, Clone, Default)]
pub struct ErrorContext {
    /// HTTP status code from the provider.
    pub http_status: Option<u16>,
    /// Provider-specific error code (e.g. "rate_limit_exceeded").
    pub provider_code: Option<String>,
    /// Request ID from provider (x-request-id header).
    pub request_id: Option<String>,
}

impl ErrorContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_status(mut self, status: u16) -> Self {
        self.http_status = Some(status);
        self
    }

    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.provider_code = Some(code.into());
        self
    }

    pub fn with_request_id(mut self, id: impl Into<String>) -> Self {
        self.request_id = Some(id.into());
        self
    }
}

/// Errors that can occur when calling providers.
#[derive(Debug, Error)]
pub enum ProviderError {
    /// Global spend guard blocked the request.
    #[error("budget exceeded: ${spend_usd:.2} >= ${limit_usd:.2}")]
    BudgetExceeded {
        limit_usd: f64,
        spend_usd: f64,
        retry_after: Duration,
    },

    /// Rate limited - caller should retry after the specified duration.
    #[error("rate limited ({limit_source:?}), retry after {retry_after:?}")]
    RateLimited {
        retry_after: Duration,
        limit_source: RateLimitSource,
        context: Option<ErrorContext>,
    },

    /// Invalid request - permanent error, don't retry.
    #[error("invalid request: {message}")]
    InvalidRequest {
        message: String,
        context: Option<ErrorContext>,
    },

    /// Provider refused the request (content policy, etc.) - permanent error.
    #[error("refused: {message}")]
    Refused {
        message: String,
        context: Option<ErrorContext>,
    },

    /// Provider error - may be retryable.
    #[error("{provider} error: {message}")]
    Provider {
        provider: &'static str,
        message: String,
        retryable: bool,
        context: Option<ErrorContext>,
    },

    /// Request timed out - retryable.
    #[error("timeout after {0:?}")]
    Timeout(Duration, Option<ErrorContext>),

    /// HTTP/network error.
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    /// Configuration error (missing API key, etc.).
    #[error("configuration error: {0}")]
    Config(String),
}

impl ProviderError {
    /// Create a budget exceeded error.
    pub fn budget_exceeded(limit_usd: f64, spend_usd: f64, retry_after: Duration) -> Self {
        Self::BudgetExceeded {
            limit_usd,
            spend_usd,
            retry_after,
        }
    }

    /// Create a rate limited error from local limiter.
    pub fn rate_limited_local(retry_after: Duration) -> Self {
        Self::RateLimited {
            retry_after,
            limit_source: RateLimitSource::Local,
            context: None,
        }
    }

    /// Create a rate limited error from remote provider.
    pub fn rate_limited_remote(retry_after: Duration, context: ErrorContext) -> Self {
        Self::RateLimited {
            retry_after,
            limit_source: RateLimitSource::Remote,
            context: Some(context),
        }
    }

    /// Create an invalid request error.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::InvalidRequest {
            message: message.into(),
            context: None,
        }
    }

    /// Create a refused error.
    pub fn refused(message: impl Into<String>) -> Self {
        Self::Refused {
            message: message.into(),
            context: None,
        }
    }

    /// Create a provider error.
    pub fn provider(provider: &'static str, message: impl Into<String>, retryable: bool) -> Self {
        Self::Provider {
            provider,
            message: message.into(),
            retryable,
            context: None,
        }
    }

    /// Create a provider error with context.
    pub fn provider_with_context(
        provider: &'static str,
        message: impl Into<String>,
        retryable: bool,
        context: ErrorContext,
    ) -> Self {
        Self::Provider {
            provider,
            message: message.into(),
            retryable,
            context: Some(context),
        }
    }

    /// Create a config error.
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config(message.into())
    }

    /// Whether this error is retryable.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::BudgetExceeded { .. } => false,
            Self::RateLimited { .. } => true,
            Self::Timeout(_, _) => true,
            Self::Provider { retryable, .. } => *retryable,
            Self::Http(e) => e.is_timeout() || e.is_connect(),
            Self::InvalidRequest { .. } => false,
            Self::Refused { .. } => false,
            Self::Config(_) => false,
        }
    }

    /// Get a short error code for logging.
    pub fn code(&self) -> &'static str {
        match self {
            Self::BudgetExceeded { .. } => "budget_exceeded",
            Self::RateLimited {
                limit_source: RateLimitSource::Local,
                ..
            } => "rate_limited_local",
            Self::RateLimited {
                limit_source: RateLimitSource::Remote,
                ..
            } => "rate_limited_remote",
            Self::InvalidRequest { .. } => "invalid_request",
            Self::Refused { .. } => "refused",
            Self::Provider { .. } => "provider_error",
            Self::Timeout(_, _) => "timeout",
            Self::Http(_) => "http_error",
            Self::Config(_) => "config_error",
        }
    }

    /// Get the error context if available.
    pub fn context(&self) -> Option<&ErrorContext> {
        match self {
            Self::BudgetExceeded { .. } => None,
            Self::RateLimited { context, .. } => context.as_ref(),
            Self::InvalidRequest { context, .. } => context.as_ref(),
            Self::Refused { context, .. } => context.as_ref(),
            Self::Provider { context, .. } => context.as_ref(),
            Self::Timeout(_, context) => context.as_ref(),
            Self::Http(_) => None,
            Self::Config(_) => None,
        }
    }

    /// Get the request ID if available.
    pub fn request_id(&self) -> Option<&str> {
        self.context().and_then(|c| c.request_id.as_deref())
    }
}
