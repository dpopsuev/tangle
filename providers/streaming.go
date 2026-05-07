package providers

import (
	"context"

	anyllm "github.com/mozilla-ai/any-llm-go/providers"
)

type StreamingProvider interface {
	CompletionWithCallback(ctx context.Context, params anyllm.CompletionParams, onToken func(string)) (*anyllm.ChatCompletion, error)
}
