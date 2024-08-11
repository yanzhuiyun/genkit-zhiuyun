package main

import (
	"context"
	"errors"
	"fmt"
	"log"

	// Import the Genkit core libraries.
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"

	// Import the Ollama (e.g. Gemma) plugin.
	"github.com/firebase/genkit/go/plugins/ollama"
)

func main() {
	ctx := context.Background()

	// Initialize the Ollama plugin.
	err := ollama.Init(ctx,
		// The address of your Ollama API server. This is often a different host
		// from your app backend (which runs Genkit), in order to run Ollama on
		// a GPU-accelerated machine.
		&ollama.Config{ServerAddress: "http://127.0.0.1:11434"})
	if err != nil {
		log.Fatal(err)
	}
	// The models you want to use. These must already be downloaded and
	// available to the Ollama server.
	ollama.DefineModel(ollama.ModelDefinition{Name: "qwen2:0.5b", Type: "CHAT"}, nil)

	// Define a simple flow that prompts an LLM to generate menu suggestions.
	genkit.DefineFlow("menuSuggestionFlow", func(ctx context.Context, input string) (string, error) {
		// Ollama provides an interface to many open generative models. Here,
		// we specify Google's Gemma model, which we configured the Ollama
		// plugin to provide, above.
		m := ollama.Model("qwen2:0.5b")
		if m == nil {
			return "", errors.New("menuSuggestionFlow: failed to find model")
		}

		// Construct a request and send it to the model API.
		resp, err := m.Generate(ctx,
			ai.NewGenerateRequest(
				&ai.GenerationCommonConfig{Temperature: 1},
				ai.NewUserTextMessage(fmt.Sprintf(`Suggest an item for the menu of a %s themed restaurant`, input))),
			nil)
		if err != nil {
			return "", err
		}

		// Handle the response from the model API. In this sample, we just
		// convert it to a string, but more complicated flows might coerce the
		// response into structured output or chain the response into another
		// LLM call, etc.
		text := resp.Text()
		return text, nil
	})

	// Initialize Genkit and start a flow server. This call must come last,
	// after all of your plug-in configuration and flow definitions. When you
	// pass a nil configuration to Init, Genkit starts a local flow server,
	// which you can interact with using the developer UI.
	if err := genkit.Init(ctx, &genkit.Options{
		FlowAddr: "localhost:9090",
	}); err != nil {
		log.Fatal(err)
	}
}
