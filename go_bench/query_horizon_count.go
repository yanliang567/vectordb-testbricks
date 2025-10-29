package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

// Count range definition
type CountRange struct {
	Min   int64
	Max   int64
	Label string
}

var countRanges = []CountRange{
	{0, 1999, "0-1999"},
	{2000, 4999, "2000-4999"},
	{5000, 9999, "5000-9999"},
	{10000, 12999, "10000-12999"},
	{13000, 14999, "13000-14999"},
	{15000, 999999999, "15000+"},
}

// FileStatistics stores statistics for one expression file
type FileStatistics struct {
	FileName          string
	TotalQueries      int
	SuccessCount      int
	FailureCount      int
	TotalCount        int64
	MinCount          int64
	MaxCount          int64
	CountValues       []int64
	RangeDistribution map[string]int
	TotalLatency      time.Duration
	FieldStats        map[string]*FieldStatistics // Statistics per field
}

// FieldStatistics stores statistics for a specific field
type FieldStatistics struct {
	FieldName         string
	SuccessCount      int
	FailureCount      int
	TotalCount        int64
	MinCount          int64
	MaxCount          int64
	CountValues       []int64
	RangeDistribution map[string]int
	TotalLatency      time.Duration
}

// NewFileStatistics creates a new FileStatistics instance
func NewFileStatistics(fileName string) *FileStatistics {
	return &FileStatistics{
		FileName:          fileName,
		MinCount:          999999999,
		MaxCount:          0,
		CountValues:       make([]int64, 0),
		RangeDistribution: make(map[string]int),
		FieldStats:        make(map[string]*FieldStatistics),
	}
}

// NewFieldStatistics creates a new FieldStatistics instance
func NewFieldStatistics(fieldName string) *FieldStatistics {
	return &FieldStatistics{
		FieldName:         fieldName,
		MinCount:          999999999,
		MaxCount:          0,
		CountValues:       make([]int64, 0),
		RangeDistribution: make(map[string]int),
	}
}

// RecordCount records a count result
func (fs *FileStatistics) RecordCount(count int64, latency time.Duration) {
	fs.SuccessCount++
	fs.TotalCount += count
	fs.CountValues = append(fs.CountValues, count)
	fs.TotalLatency += latency

	if count < fs.MinCount {
		fs.MinCount = count
	}
	if count > fs.MaxCount {
		fs.MaxCount = count
	}

	// Classify into range
	rangeLabel := getRangeLabel(count)
	fs.RangeDistribution[rangeLabel]++
}

// RecordFailure records a failed query
func (fs *FileStatistics) RecordFailure() {
	fs.FailureCount++
}

// RecordCountForField records a count result for a specific field
func (fs *FileStatistics) RecordCountForField(fieldName string, count int64, latency time.Duration) {
	// Create field stats if not exists
	if _, exists := fs.FieldStats[fieldName]; !exists {
		fs.FieldStats[fieldName] = NewFieldStatistics(fieldName)
	}

	fieldStats := fs.FieldStats[fieldName]
	fieldStats.SuccessCount++
	fieldStats.TotalCount += count
	fieldStats.CountValues = append(fieldStats.CountValues, count)
	fieldStats.TotalLatency += latency

	if count < fieldStats.MinCount {
		fieldStats.MinCount = count
	}
	if count > fieldStats.MaxCount {
		fieldStats.MaxCount = count
	}

	rangeLabel := getRangeLabel(count)
	fieldStats.RangeDistribution[rangeLabel]++
}

// RecordFailureForField records a failed query for a specific field
func (fs *FileStatistics) RecordFailureForField(fieldName string) {
	if _, exists := fs.FieldStats[fieldName]; !exists {
		fs.FieldStats[fieldName] = NewFieldStatistics(fieldName)
	}
	fs.FieldStats[fieldName].FailureCount++
}

// getRangeLabel returns the range label for a count value
func getRangeLabel(count int64) string {
	for _, r := range countRanges {
		if count >= r.Min && count <= r.Max {
			return r.Label
		}
	}
	return "Unknown"
}

// PrintStatistics prints detailed statistics for this file
func (fs *FileStatistics) PrintStatistics() {
	log.Printf("\n%s", strings.Repeat("=", 100))
	log.Printf("📊 Statistics: %s", fs.FileName)
	log.Printf("%s", strings.Repeat("=", 100))

	log.Printf("\n   Summary:")
	log.Printf("      • Total queries: %d", fs.TotalQueries)
	log.Printf("      • Success: %d", fs.SuccessCount)
	log.Printf("      • Failed: %d", fs.FailureCount)

	if fs.SuccessCount > 0 {
		log.Printf("      • Total count: %d", fs.TotalCount)
		log.Printf("      • Average count: %.2f", float64(fs.TotalCount)/float64(fs.SuccessCount))
		log.Printf("      • Min count: %d", fs.MinCount)
		log.Printf("      • Max count: %d", fs.MaxCount)
		log.Printf("      • Average latency: %v", fs.TotalLatency/time.Duration(fs.SuccessCount))
	}

	// Print overall range distribution if no field-level stats
	if len(fs.FieldStats) == 0 {
		log.Printf("\n   Count Distribution:")
		log.Printf("      %-15s  %6s  %7s  %s", "Range", "Count", "Percent", "Distribution")
		log.Printf("      %s  %s  %s  %s", strings.Repeat("-", 15), strings.Repeat("-", 6), strings.Repeat("-", 7), strings.Repeat("-", 50))

		for _, r := range countRanges {
			count := fs.RangeDistribution[r.Label]
			percentage := 0.0
			if fs.SuccessCount > 0 {
				percentage = float64(count) / float64(fs.SuccessCount) * 100
			}
			barLength := int(percentage / 2) // 2% per character
			bar := strings.Repeat("█", barLength)
			log.Printf("      %-15s  %6d  %6.1f%%  %s", r.Label, count, percentage, bar)
		}
	}

	// Print per-field statistics if available
	if len(fs.FieldStats) > 0 {
		log.Printf("\n   Statistics by Field:")

		// Sort field names for consistent output
		var fieldNames []string
		for fieldName := range fs.FieldStats {
			fieldNames = append(fieldNames, fieldName)
		}
		sort.Strings(fieldNames)

		for _, fieldName := range fieldNames {
			fieldStat := fs.FieldStats[fieldName]
			log.Printf("\n      🔹 Field: %s", fieldName)
			log.Printf("         • Success queries: %d", fieldStat.SuccessCount)
			log.Printf("         • Failed queries: %d", fieldStat.FailureCount)

			if fieldStat.SuccessCount > 0 {
				log.Printf("         • Total count: %d", fieldStat.TotalCount)
				log.Printf("         • Average count: %.2f", float64(fieldStat.TotalCount)/float64(fieldStat.SuccessCount))
				log.Printf("         • Min count: %d", fieldStat.MinCount)
				log.Printf("         • Max count: %d", fieldStat.MaxCount)
				log.Printf("         • Average latency: %v", fieldStat.TotalLatency/time.Duration(fieldStat.SuccessCount))

				// Print range distribution for this field
				log.Printf("         • Count distribution:")
				for _, r := range countRanges {
					count := fieldStat.RangeDistribution[r.Label]
					if count > 0 {
						percentage := float64(count) / float64(fieldStat.SuccessCount) * 100
						barLength := int(percentage / 2)
						bar := strings.Repeat("█", barLength)
						log.Printf("           %-15s  %6d  %6.1f%%  %s", r.Label, count, percentage, bar)
					}
				}
			}
		}
	}

	log.Printf("%s\n", strings.Repeat("=", 100))
}

// expandPath expands ~ to user home directory
func expandPath(path string) (string, error) {
	if strings.HasPrefix(path, "~/") {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("failed to get home directory: %v", err)
		}
		return filepath.Join(homeDir, path[2:]), nil
	}
	return path, nil
}

// splitAndExpressions splits an expression by AND operators, respecting parentheses
func splitAndExpressions(expression string) []string {
	// Simple approach: split by " and " while respecting parentheses
	var result []string
	var depth int
	var current strings.Builder

	i := 0
	for i < len(expression) {
		char := expression[i]

		if char == '(' {
			depth++
			current.WriteByte(char)
			i++
		} else if char == ')' {
			depth--
			current.WriteByte(char)
			i++
		} else if depth == 0 && i+5 <= len(expression) && strings.ToLower(expression[i:i+5]) == " and " {
			// Found an AND at depth 0
			if current.Len() > 0 {
				result = append(result, strings.TrimSpace(current.String()))
				current.Reset()
			}
			i += 5 // Skip " and "
		} else {
			current.WriteByte(char)
			i++
		}
	}

	if current.Len() > 0 {
		result = append(result, strings.TrimSpace(current.String()))
	}

	return result
}

// extractFieldName extracts the field name from a simple expression
func extractFieldName(expression string) string {
	// Remove leading/trailing spaces
	expr := strings.TrimSpace(expression)

	// Pattern 1: ARRAY_CONTAINS_ANY(field_name, [...])
	// Pattern 2: ARRAY_CONTAINS_ALL(field_name, [...])
	arrayFuncRe := regexp.MustCompile(`^ARRAY_CONTAINS_(?:ANY|ALL)\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*,`)
	if matches := arrayFuncRe.FindStringSubmatch(expr); len(matches) > 1 {
		return matches[1]
	}

	// Remove leading/trailing parentheses for other patterns
	expr = strings.Trim(expr, "()")
	expr = strings.TrimSpace(expr)

	// Pattern 3: field_name IN [...] or field_name in [...]
	// Pattern 4: field_name == value, field_name >= value, etc.
	// Pattern 5: field_name like "..."

	// Use regex to extract field name (word characters before an operator)
	// Note: Added IN (uppercase) to the pattern
	re := regexp.MustCompile(`^([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:==|!=|>=|<=|>|<|[Ii][Nn]\s|[Ll][Ii][Kk][Ee]\s)`)
	matches := re.FindStringSubmatch(expr)
	if len(matches) > 1 {
		return matches[1]
	}

	return ""
}

// filterExpressionsByFieldName filters sub-expressions that contain the specified field name
func filterExpressionsByFieldName(subExpressions []string, fieldName string) []string {
	var result []string
	for _, expr := range subExpressions {
		extractedField := extractFieldName(expr)
		if extractedField == fieldName {
			result = append(result, expr)
		}
	}
	return result
}

// groupExpressionsByField groups sub-expressions by their field names
func groupExpressionsByField(subExpressions []string) map[string][]string {
	result := make(map[string][]string)
	for _, expr := range subExpressions {
		fieldName := extractFieldName(expr)
		if fieldName != "" {
			result[fieldName] = append(result[fieldName], expr)
		}
	}
	return result
}

// mergeCoordinateFields merges lat/lon pairs into a single group
// For example, gcj02_lat and gcj02_lon will be merged into "gcj02_lat_lon"
func mergeCoordinateFields(fieldGroups map[string][]string) map[string][]string {
	result := make(map[string][]string)
	processed := make(map[string]bool)

	for fieldName, exprs := range fieldGroups {
		if processed[fieldName] {
			continue
		}

		// Check if this is a latitude field
		if strings.HasSuffix(fieldName, "_lat") {
			// Look for corresponding longitude field
			baseField := strings.TrimSuffix(fieldName, "_lat")
			lonField := baseField + "_lon"

			if lonExprs, exists := fieldGroups[lonField]; exists {
				// Merge lat and lon into one group
				mergedField := baseField + "_lat_lon"
				mergedExprs := append([]string{}, exprs...)
				mergedExprs = append(mergedExprs, lonExprs...)
				result[mergedField] = mergedExprs
				processed[fieldName] = true
				processed[lonField] = true
				continue
			}
		}

		// Check if this is a longitude field without corresponding latitude
		if strings.HasSuffix(fieldName, "_lon") {
			baseField := strings.TrimSuffix(fieldName, "_lon")
			latField := baseField + "_lat"

			if _, exists := fieldGroups[latField]; exists {
				// Already processed in the lat case above
				continue
			}
		}

		// Not a coordinate pair, add as-is
		result[fieldName] = exprs
	}

	return result
}

// PreGeneratedQuery represents a query with pre-generated expression
type PreGeneratedQuery struct {
	QueryID          int                    `json:"query_id"`
	OriginalQuery    map[string]interface{} `json:"original_query"`
	MilvusExpression string                 `json:"milvus_expression"`
}

// ExpressionFileData represents the structure of expression JSON files
type ExpressionFileData struct {
	SourceFile   string              `json:"source_file"`
	TotalQueries int                 `json:"total_queries"`
	Queries      []PreGeneratedQuery `json:"queries"`
}

// CountResult represents the result of a count query
type CountResult struct {
	QueryID    int
	Expression string
	Count      int64
	Latency    time.Duration
	Error      string
}

// QueryCountTester performs count queries for each expression
type QueryCountTester struct {
	client     *milvusclient.Client
	collection string
	condition  string
}

// NewQueryCountTester creates a new count query tester
func NewQueryCountTester(uri, token, collectionName, condition string) (*QueryCountTester, error) {
	ctx := context.Background()

	// Create Milvus client
	clientConfig := &milvusclient.ClientConfig{
		Address: uri,
	}

	if token != "" {
		clientConfig.APIKey = token
	}

	milvusClient, err := milvusclient.New(ctx, clientConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create Milvus client: %v", err)
	}

	// Verify collection exists and is loaded
	has, err := milvusClient.HasCollection(ctx, milvusclient.NewHasCollectionOption(collectionName))
	if err != nil {
		return nil, fmt.Errorf("failed to check collection '%s' existence: %v", collectionName, err)
	}
	if !has {
		return nil, fmt.Errorf("collection '%s' does not exist", collectionName)
	}

	// Check if collection is loaded
	loadState, err := milvusClient.GetLoadState(ctx, milvusclient.NewGetLoadStateOption(collectionName))
	if err != nil {
		return nil, fmt.Errorf("failed to get load state: %v", err)
	}

	if loadState.State != entity.LoadStateLoaded {
		return nil, fmt.Errorf("collection %s is not loaded (current state: %v)", collectionName, loadState.State)
	}

	return &QueryCountTester{
		client:     milvusClient,
		collection: collectionName,
		condition:  condition,
	}, nil
}

// LoadExpressionFile loads expressions from a JSON file
func LoadExpressionFile(filePath string) (*ExpressionFileData, error) {
	// Expand ~ path to full path
	expandedPath, err := expandPath(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to expand file path %s: %v", filePath, err)
	}

	// Check if file exists
	if _, err := os.Stat(expandedPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("❌ Expression file does not exist: %s", expandedPath)
	}

	log.Printf("📖 Loading expressions from %s", expandedPath)

	file, err := os.Open(expandedPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open expression file: %v", err)
	}
	defer file.Close()

	// Read and parse JSON
	data, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read expression file: %v", err)
	}

	var exprData ExpressionFileData
	if err := json.Unmarshal(data, &exprData); err != nil {
		return nil, fmt.Errorf("failed to parse expression JSON: %v", err)
	}

	if len(exprData.Queries) == 0 {
		return nil, fmt.Errorf("❌ No queries found in %s", filePath)
	}

	log.Printf("✅ Loaded %d expressions from %s", len(exprData.Queries), filepath.Base(expandedPath))
	return &exprData, nil
}

// CountForExpression executes a count query for a single expression
func (qct *QueryCountTester) CountForExpression(ctx context.Context, queryID int, expression string, queryTimeout time.Duration) *CountResult {
	startTime := time.Now()

	queryCtx, cancel := context.WithTimeout(ctx, queryTimeout)
	defer cancel()

	result := &CountResult{
		QueryID:    queryID,
		Expression: expression,
	}

	// Execute query with count(*)
	queryOption := milvusclient.NewQueryOption(qct.collection).
		WithFilter(expression).
		WithOutputFields("count(*)")

	queryResult, err := qct.client.Query(queryCtx, queryOption)
	result.Latency = time.Since(startTime)

	if err != nil {
		result.Error = fmt.Sprintf("query failed: %v", err)
		return result
	}

	// Extract count from result
	if queryResult.ResultCount > 0 {
		// The count(*) result is in the output fields
		for _, field := range queryResult.Fields {
			if field.Name() == "count(*)" {
				// Get the first value (should be the count)
				if intCol, ok := field.(*column.ColumnInt64); ok {
					if intCol.Len() > 0 {
						count, err := intCol.Get(0)
						if err == nil {
							if countVal, ok := count.(int64); ok {
								result.Count = countVal
							}
						}
					}
				}
				break
			}
		}
	}

	return result
}

// RunCountTest runs count queries for all expressions in a file
func (qct *QueryCountTester) RunCountTest(ctx context.Context, exprFile string, queryTimeout time.Duration) (*FileStatistics, error) {
	// Load expression file
	exprData, err := LoadExpressionFile(exprFile)
	if err != nil {
		return nil, err
	}

	// Create statistics
	stats := NewFileStatistics(filepath.Base(exprFile))
	stats.TotalQueries = len(exprData.Queries)

	log.Printf("\n🚀 Processing: %s", filepath.Base(exprFile))
	log.Printf("   Collection: %s", qct.collection)
	log.Printf("   Condition mode: %s", qct.condition)
	log.Printf("   Total queries: %d", len(exprData.Queries))
	log.Printf("   Query timeout: %v", queryTimeout)

	// Execute count query for each expression
	for i, query := range exprData.Queries {
		expression := query.MilvusExpression

		// Show progress every 50 queries
		if i%50 == 0 || i == len(exprData.Queries)-1 {
			log.Printf("   Progress: %d/%d queries completed", i+1, len(exprData.Queries))
		}

		// Process based on condition mode
		switch qct.condition {
		case "entire":
			// Mode 1: Count the entire expression as is
			result := qct.CountForExpression(ctx, query.QueryID, expression, queryTimeout)
			qct.logAndRecordResult(i, len(exprData.Queries), result, expression, stats)

		case "each":
			// Mode 2: Group by field and count merged expressions for each field
			subExpressions := splitAndExpressions(expression)
			fieldGroups := groupExpressionsByField(subExpressions)

			// Merge coordinate fields (lat/lon pairs)
			fieldGroups = mergeCoordinateFields(fieldGroups)

			for fieldName, fieldExprs := range fieldGroups {
				// Merge expressions for this field
				mergedExpression := strings.Join(fieldExprs, " and ")
				result := qct.CountForExpression(ctx, query.QueryID, mergedExpression, queryTimeout)
				qct.logAndRecordResultForField(i, len(exprData.Queries), result, fieldName, mergedExpression, stats)
				time.Sleep(5 * time.Millisecond)
			}

		default:
			// Mode 3: Filter by field name and merge matching sub-expressions with AND
			subExpressions := splitAndExpressions(expression)
			filteredExpressions := filterExpressionsByFieldName(subExpressions, qct.condition)

			if len(filteredExpressions) > 0 {
				// Merge all filtered expressions with AND
				mergedExpression := strings.Join(filteredExpressions, " and ")
				// Execute count for the merged expression
				result := qct.CountForExpression(ctx, query.QueryID, mergedExpression, queryTimeout)
				qct.logAndRecordResult(i, len(exprData.Queries), result, mergedExpression, stats)
			}
		}

		// Brief pause between queries to avoid overwhelming the server
		time.Sleep(10 * time.Millisecond)
	}

	return stats, nil
}

// logAndRecordResult logs and records a count result (minimal logging)
func (qct *QueryCountTester) logAndRecordResult(idx, total int, result *CountResult, expression string, stats *FileStatistics) {
	if result.Error != "" {
		stats.RecordFailure()
	} else {
		stats.RecordCount(result.Count, result.Latency)
	}
}

// logAndRecordResultForField logs and records a count result for a specific field (minimal logging)
func (qct *QueryCountTester) logAndRecordResultForField(idx, total int, result *CountResult, fieldName, expression string, stats *FileStatistics) {
	if result.Error != "" {
		stats.RecordFailureForField(fieldName)
	} else {
		stats.RecordCountForField(fieldName, result.Count, result.Latency)
	}
}

// Close closes the Milvus client connection
func (qct *QueryCountTester) Close() error {
	ctx := context.Background()
	return qct.client.Close(ctx)
}

// PrintSummaryTable prints a summary comparison table for all files
func PrintSummaryTable(allStats []*FileStatistics) {
	log.Printf("\n%s", strings.Repeat("=", 100))
	log.Printf("📊 Count Distribution Comparison")
	log.Printf("%s", strings.Repeat("=", 100))

	// Header
	header := fmt.Sprintf("\n%-25s", "File")
	for _, r := range countRanges {
		header += fmt.Sprintf(" %13s", r.Label)
	}
	header += fmt.Sprintf(" %10s", "Total")
	log.Printf("%s", header)
	log.Printf("%s", strings.Repeat("-", 100))

	// Each file
	for _, stats := range allStats {
		displayName := strings.Replace(stats.FileName, "_exprs.json", "", 1)
		displayName = strings.Replace(displayName, "_geo", "(geo)", 1)

		row := fmt.Sprintf("%-25s", displayName)
		for _, r := range countRanges {
			count := stats.RangeDistribution[r.Label]
			row += fmt.Sprintf(" %13d", count)
		}
		row += fmt.Sprintf(" %10d", stats.TotalQueries)
		log.Printf("%s", row)
	}

	// Total row
	log.Printf("%s", strings.Repeat("-", 100))
	totalRow := fmt.Sprintf("%-25s", "Total")
	totalQueries := 0
	for _, r := range countRanges {
		totalInRange := 0
		for _, stats := range allStats {
			totalInRange += stats.RangeDistribution[r.Label]
		}
		totalRow += fmt.Sprintf(" %13d", totalInRange)
	}
	for _, stats := range allStats {
		totalQueries += stats.TotalQueries
	}
	totalRow += fmt.Sprintf(" %10d", totalQueries)
	log.Printf("%s", totalRow)

	log.Printf("%s\n", strings.Repeat("=", 100))
}

func main() {
	// Command line arguments
	var (
		exprFiles       = flag.String("expr-files", "", "Comma-separated expression file paths or 'all' for all files in expr-dir")
		exprDir         = flag.String("expr-dir", "/root/horizon/horizonPoc/data/query_expressions", "Directory containing expression JSON files")
		filePattern     = flag.String("file-pattern", "*_exprs.json", "File pattern to match when using 'all' (e.g., '*_exprs.json', '*txt.json', '*.json')")
		queryTimeoutSec = flag.Int("query-timeout", 30, "Individual query timeout in seconds")
		condition       = flag.String("condition", "entire", "Condition mode: 'entire' (count entire expression), 'each' (count each AND sub-expression), or a field name (count only expressions with that field)")

		// Hardcoded values
		host           = "https://in01-3e1accccxxxx8817d.ali-cn-hangzhou.cloud-uat.zilliz.cn:19530"
		collectionName = "horizon_test_collection"
		apiKey         = "token"
	)

	flag.Parse()

	log.SetFlags(log.LstdFlags | log.Lshortfile)

	// Check if expr-files is provided
	if *exprFiles == "" {
		log.Fatalf("❌ --expr-files parameter is required. Example: 'qc_1_exprs.json' or 'all'")
	}

	// Validate condition parameter
	if *condition == "" {
		log.Fatalf("❌ --condition parameter cannot be empty. Use 'entire', 'each', or a field name.")
	}

	log.Printf("📋 Configuration:")
	log.Printf("   Condition mode: %s", *condition)
	if *condition == "entire" {
		log.Printf("   → Count entire expressions")
	} else if *condition == "each" {
		log.Printf("   → Group by field and count merged expressions")
	} else {
		log.Printf("   → Count only field: %s", *condition)
	}
	if *exprFiles == "all" {
		log.Printf("   File pattern: %s", *filePattern)
	}

	// Get expression directory path
	exprDirPath, err := expandPath(*exprDir)
	if err != nil {
		log.Fatalf("❌ Failed to expand expression directory path: %v", err)
	}

	// Parse expression files
	var expressionFiles []string

	if *exprFiles == "all" {
		// Load all expression files from directory matching the pattern
		pattern := filepath.Join(exprDirPath, *filePattern)
		matches, err := filepath.Glob(pattern)
		if err != nil {
			log.Fatalf("❌ Failed to match pattern '%s': %v", pattern, err)
		}

		// Filter out directories, keep only files
		for _, match := range matches {
			fileInfo, err := os.Stat(match)
			if err != nil {
				log.Printf("⚠️  Warning: Cannot stat file %s: %v", match, err)
				continue
			}
			if !fileInfo.IsDir() {
				expressionFiles = append(expressionFiles, match)
			}
		}

		if len(expressionFiles) == 0 {
			log.Fatalf("❌ No expression files found matching pattern '%s' in %s", *filePattern, exprDirPath)
		}

		log.Printf("✅ Found %d files matching pattern '%s'", len(expressionFiles), *filePattern)
	} else {
		// Parse comma-separated file list
		fileList := strings.Split(*exprFiles, ",")
		for _, fileName := range fileList {
			fileName = strings.TrimSpace(fileName)
			if fileName == "" {
				continue
			}

			var fullPath string
			if filepath.IsAbs(fileName) {
				fullPath = fileName
			} else {
				fullPath = filepath.Join(exprDirPath, fileName)
			}

			// Check if file exists
			if _, err := os.Stat(fullPath); os.IsNotExist(err) {
				log.Fatalf("❌ Expression file does not exist: %s", fullPath)
			}

			expressionFiles = append(expressionFiles, fullPath)
		}
	}

	log.Printf("📋 Files to process (%d):", len(expressionFiles))
	for i, f := range expressionFiles {
		log.Printf("   %d. %s", i+1, filepath.Base(f))
	}

	// Create count query tester
	qct, err := NewQueryCountTester(host, apiKey, collectionName, *condition)
	if err != nil {
		log.Fatalf("❌ Failed to create count query tester: %v", err)
	}
	defer qct.Close()

	ctx := context.Background()
	queryTimeout := time.Duration(*queryTimeoutSec) * time.Second

	// Store all statistics for summary table
	var allStats []*FileStatistics

	// Process each expression file
	for i, exprFile := range expressionFiles {
		log.Printf("\n\n🔍 Processing file %d/%d: %s", i+1, len(expressionFiles), filepath.Base(exprFile))

		stats, err := qct.RunCountTest(ctx, exprFile, queryTimeout)
		if err != nil {
			log.Printf("❌ Failed to process %s: %v", filepath.Base(exprFile), err)
			continue
		}

		// Print statistics for this file
		stats.PrintStatistics()

		// Store for summary
		allStats = append(allStats, stats)

		// Brief pause between files
		if i < len(expressionFiles)-1 {
			time.Sleep(2 * time.Second)
		}
	}

	// Print summary table
	if len(allStats) > 0 {
		PrintSummaryTable(allStats)
	}

	log.Printf("\n\n✅ All files processed successfully!")
}
