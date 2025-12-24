package main

import (
	"context"
	"fmt"
	"math/rand"
	"os"

	"github.com/openai/openai-go"
	"github.com/turbopuffer/turbopuffer-go"
	"github.com/turbopuffer/turbopuffer-go/option"
)

// Create an embedding with OpenAI, could be {Cohere, Voyage, Mixed Bread, ...}
// Requires OPENAI_API_KEY to be set (https://platform.openai.com/settings/organization/api-keys)
func openaiOrRandVector(ctx context.Context, text string) []float32 {
	if os.Getenv("OPENAI_API_KEY") == "" {
		fmt.Println("OPENAI_API_KEY not set, using random vectors")
		return []float32{rand.Float32(), rand.Float32()}
	}

	client := openai.NewClient()
	resp, err := client.Embeddings.New(ctx, openai.EmbeddingNewParams{
		Input: openai.EmbeddingNewParamsInputUnion{OfString: openai.String(text)},
		Model: openai.EmbeddingModelTextEmbedding3Small,
	})
	if err != nil {
		fmt.Printf("OpenAI error, using random vectors: %v\n", err)
		return []float32{rand.Float32(), rand.Float32()}
	}
	embedding := make([]float32, len(resp.Data[0].Embedding))
	for i, v := range resp.Data[0].Embedding {
		embedding[i] = float32(v)
	}
	return embedding
}

func main() {
	ctx := context.Background()
	tpuf := turbopuffer.NewClient(
		// API tokens are created in the dashboard: https://turbopuffer.com/dashboard
		option.WithAPIKey(os.Getenv("TURBOPUFFER_API_KEY")),
		// Pick the right region: https://turbopuffer.com/docs/regions
		option.WithRegion("gcp-us-central1"),
	)

	ns := tpuf.Namespace("vector-1-example-go")

	_, err := ns.Write(
		ctx,
		turbopuffer.NamespaceWriteParams{
			UpsertRows: []turbopuffer.RowParam{
				{
					"id":       1,
					"vector":   openaiOrRandVector(ctx, "A cat sleeping on a windowsill"),
					"text":     "A cat sleeping on a windowsill",
					"category": "animal",
				},
				{
					"id":       2,
					"vector":   openaiOrRandVector(ctx, "A playful kitten chasing a toy"),
					"text":     "A playful kitten chasing a toy",
					"category": "animal",
				},
				{
					"id":       3,
					"vector":   openaiOrRandVector(ctx, "An airplane flying through clouds"),
					"text":     "An airplane flying through clouds",
					"category": "vehicle",
				},
			},
			DistanceMetric: turbopuffer.DistanceMetricCosineDistance,
		},
	)
	if err != nil {
		panic(err)
	}

	// Basic vector search
	result, err := ns.Query(
		ctx,
		turbopuffer.NamespaceQueryParams{
			RankBy:            turbopuffer.NewRankByVector("vector", openaiOrRandVector(ctx, "feline")),
			TopK:              turbopuffer.Int(2),
			IncludeAttributes: turbopuffer.IncludeAttributesParam{StringArray: []string{"text"}},
		},
	)
	if err != nil {
		panic(err)
	}
	// Returns cat and kitten documents, sorted by vector similarity
	fmt.Print(turbopuffer.PrettyPrint(result.Rows))

	// Example of vector search with filters
	ns2 := tpuf.Namespace("vector-2-example-go")
	_, err = ns2.Write(
		ctx,
		turbopuffer.NamespaceWriteParams{
			UpsertRows: []turbopuffer.RowParam{
				{
					"id":          1,
					"vector":      openaiOrRandVector(ctx, "A shiny red sports car"),
					"description": "A shiny red sports car",
					"color":       "red",
					"type":        "car",
					"price":       50000,
				},
				{
					"id":          2,
					"vector":      openaiOrRandVector(ctx, "A sleek blue sedan"),
					"description": "A sleek blue sedan",
					"color":       "blue",
					"type":        "car",
					"price":       35000,
				},
				{
					"id":          3,
					"vector":      openaiOrRandVector(ctx, "A large red delivery truck"),
					"description": "A large red delivery truck",
					"color":       "red",
					"type":        "truck",
					"price":       80000,
				},
				{
					"id":          4,
					"vector":      openaiOrRandVector(ctx, "A blue pickup truck"),
					"description": "A blue pickup truck",
					"color":       "blue",
					"type":        "truck",
					"price":       45000,
				},
			},
			DistanceMetric: turbopuffer.DistanceMetricCosineDistance,
		},
	)
	if err != nil {
		panic(err)
	}

	result, err = ns2.Query(
		ctx,
		turbopuffer.NamespaceQueryParams{
			RankBy: turbopuffer.NewRankByVector("vector", openaiOrRandVector(ctx, "car")), // Embedding similar to "car"
			TopK:   turbopuffer.Int(10),
			// Complex filter combining multiple conditions, see https://turbopuffer.com/docs/query for all options
			Filters: turbopuffer.NewFilterAnd([]turbopuffer.Filter{
				turbopuffer.NewFilterEq("color", "blue"),
				turbopuffer.NewFilterLt("price", 40000),
				turbopuffer.NewFilterEq("type", "car"),
			}),
			IncludeAttributes: turbopuffer.IncludeAttributesParam{StringArray: []string{"description", "price"}},
		},
	)
	if err != nil {
		panic(err)
	}
	// Returns only blue cars under $40k, sorted by similarity to the query vector
	fmt.Print(turbopuffer.PrettyPrint(result.Rows))
}
