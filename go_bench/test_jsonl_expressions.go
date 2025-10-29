package main

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"
)

// Test program to verify JSONL query parsing and expression generation
func testMain() {
	// Test cases from different query condition files
	testCases := []string{
		// query_condition_1.jsonl format
		`{"timestamp": {"$gte": 1730458317213, "$lte": 1730867875654}, "type_model": "IDX.2", "expert_collected": false}`,
		`{"timestamp": {"$gte": 1733058769808, "$lte": 1733468328249}, "type_model": "IDX.9", "expert_collected": true}`,

		// query_condition_2.jsonl format
		`{"timestamp": {"$gte": 1728676386641, "$lte": 1733181529498}, "device_id": {"$in": ["DV181", "DV125", "DV282"]}, "tag_id": {"$in": ["68cd37349f89f5b6340db_183", "68cd37349f89f5b6340db_142", "68cd37349f89f5b6340db_118"]}}`,

		// query_condition_3.jsonl format
		`{"timestamp": {"$gte": 1728699802843, "$lte": 1729708954843}, "sensor_lidar_type": {"$in": ["Pandar128", "AT256", "ATX128"], "$not_in": ["AT128"]}, "tag_id": {"$in": ["68cd37349f89f5b6340db_120", "68cd37349f89f5b6340db_146"], "contains_all": ["68cd37349f89f5b6340db_177", "68cd37349f89f5b6340db_11"]}, "longitude": {"$gte": 114.88012614617554, "$lte": 125.43142425804291}, "latitude": {"$gte": 25.849973018971188, "$lte": 34.903338374260905}}`,
	}

	fmt.Println("=== Testing JSONL Query to Milvus Expression Conversion ===\n")

	for i, testCase := range testCases {
		fmt.Printf("Test Case %d:\n", i+1)
		fmt.Printf("JSON: %s\n\n", testCase)

		var qc QueryCondition
		if err := json.Unmarshal([]byte(testCase), &qc); err != nil {
			log.Printf("❌ Failed to parse JSON: %v\n", err)
			continue
		}

		expr := QueryConditionToExpression(qc)
		fmt.Printf("Milvus Expression:\n%s\n", expr)
		fmt.Println("\n" + strings.Repeat("=", 80) + "\n")
	}

	fmt.Println("✅ All test cases completed!")
}
