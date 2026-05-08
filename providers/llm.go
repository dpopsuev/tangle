package providers

import (
	"context"
	"encoding/json"
	"fmt"

	tangle "github.com/dpopsuev/tangle"
	anyllm "github.com/mozilla-ai/any-llm-go/providers"
)

type UsageRecorder func(model string, usage *anyllm.Usage)

func NewCompleter(provider anyllm.Provider, model string, recorder UsageRecorder) tangle.CompleteFunc {
	return func(ctx context.Context, params tangle.CompletionParams) (*tangle.Completion, error) {
		var msgs []anyllm.Message
		if len(params.Messages) > 0 {
			for _, m := range params.Messages {
				msg := anyllm.Message{
					Role:       m.Role,
					Content:    m.Content,
					ToolCallID: m.ToolCallID,
				}
				for _, tc := range m.ToolCalls {
					msg.ToolCalls = append(msg.ToolCalls, anyllm.ToolCall{
						ID:   tc.ID,
						Type: "function",
						Function: anyllm.FunctionCall{
							Name:      tc.Name,
							Arguments: string(tc.Input),
						},
					})
				}
				msgs = append(msgs, msg)
			}
		} else {
			msgs = []anyllm.Message{{Role: "user", Content: params.Prompt}}
		}
		req := anyllm.CompletionParams{
			Model:    model,
			Messages: msgs,
		}

		if params.MaxTokens > 0 {
			req.MaxTokens = &params.MaxTokens
		}

		for _, t := range params.Tools {
			props := make(map[string]any)
			if t.InputSchema != nil {
				json.Unmarshal(t.InputSchema, &props)
			}
			req.Tools = append(req.Tools, anyllm.Tool{
				Type: "function",
				Function: anyllm.Function{
					Name:        t.Name,
					Description: t.Description,
					Parameters:  props,
				},
			})
		}

		if params.ThinkingLevel != "" {
			if tp, ok := provider.(ThinkingProvider); ok {
				budgets := map[string]int64{
					"minimal": 128,
					"low":     512,
					"medium":  2048,
					"high":    8192,
				}
				if b, ok := budgets[params.ThinkingLevel]; ok {
					tp.SetThinkingBudget(b)
				}
			}
		}

		var resp *anyllm.ChatCompletion
		var err error

		if params.OnToken != nil {
			if sp, ok := provider.(StreamingProvider); ok {
				resp, err = sp.CompletionWithCallback(ctx, req, params.OnToken)
			} else {
				resp, err = provider.Completion(ctx, req)
			}
		} else {
			resp, err = provider.Completion(ctx, req)
		}
		if err != nil {
			return nil, fmt.Errorf("llm completion: %w", err)
		}
		if len(resp.Choices) == 0 {
			return nil, ErrNoChoices
		}

		if recorder != nil && resp.Usage != nil {
			recorder(resp.Model, resp.Usage)
		}

		completion := &tangle.Completion{}

		if content, ok := resp.Choices[0].Message.Content.(string); ok {
			completion.Content = content
		}

		for _, tc := range resp.Choices[0].Message.ToolCalls {
			input := json.RawMessage(tc.Function.Arguments)
			completion.ToolCalls = append(completion.ToolCalls, tangle.ToolCall{
				ID:    tc.ID,
				Name:  tc.Function.Name,
				Input: input,
			})
		}

		if resp.Usage != nil {
			completion.Tokens = tangle.TokenUsage{
				Input:  resp.Usage.PromptTokens,
				Output: resp.Usage.CompletionTokens,
			}
		}

		return completion, nil
	}
}

func NewCompleterFromEnv(envName, model string) (tangle.Completer, error) {
	p, err := NewProviderFromEnv(envName)
	if err != nil {
		return nil, err
	}
	return NewCompleter(p, model, nil), nil
}

func NewCompleterByName(providerName, model string) (tangle.Completer, error) {
	p, err := NewProviderByName(providerName)
	if err != nil {
		return nil, err
	}
	return NewCompleter(p, model, nil), nil
}
