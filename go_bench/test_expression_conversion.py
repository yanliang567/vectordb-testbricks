#!/usr/bin/env python3
"""
Test script to verify JSONL query to Milvus Expression conversion
"""

import json


def convert_query_to_expression(query):
    """Convert JSON query to Milvus expression"""
    conditions = []
    
    # Handle timestamp
    if 'timestamp' in query:
        if '$gte' in query['timestamp']:
            conditions.append(f"timestamp >= {int(query['timestamp']['$gte'])}")
        if '$lte' in query['timestamp']:
            conditions.append(f"timestamp <= {int(query['timestamp']['$lte'])}")
    
    # Handle type_model
    if 'type_model' in query:
        conditions.append(f'type_model == "{query["type_model"]}"')
    
    # Handle expert_collected
    if 'expert_collected' in query:
        value = "true" if query['expert_collected'] else "false"
        conditions.append(f'expert_collected == {value}')
    
    # Handle device_id
    if 'device_id' in query and '$in' in query['device_id']:
        device_ids = '","'.join(query['device_id']['$in'])
        conditions.append(f'device_id in ["{device_ids}"]')
    
    # Handle tag_id
    if 'tag_id' in query:
        if '$in' in query['tag_id']:
            tag_ids = '","'.join(query['tag_id']['$in'])
            conditions.append(f'ARRAY_CONTAINS_ANY(tag_id, ["{tag_ids}"])')
        
        if 'contains_all' in query['tag_id']:
            tag_ids = '","'.join(query['tag_id']['contains_all'])
            conditions.append(f'ARRAY_CONTAINS_ALL(tag_id, ["{tag_ids}"])')
    
    # Handle sensor_lidar_type
    if 'sensor_lidar_type' in query:
        if '$in' in query['sensor_lidar_type']:
            sensors = '","'.join(query['sensor_lidar_type']['$in'])
            conditions.append(f'ARRAY_CONTAINS_ANY(sensor_lidar_type, ["{sensors}"])')
        
        if '$not_in' in query['sensor_lidar_type']:
            for sensor in query['sensor_lidar_type']['$not_in']:
                conditions.append(f'not ARRAY_CONTAINS(sensor_lidar_type, "{sensor}")')
    
    # Handle longitude
    if 'longitude' in query:
        if '$gte' in query['longitude']:
            conditions.append(f"longitude >= {query['longitude']['$gte']}")
        if '$lte' in query['longitude']:
            conditions.append(f"longitude <= {query['longitude']['$lte']}")
    
    # Handle latitude
    if 'latitude' in query:
        if '$gte' in query['latitude']:
            conditions.append(f"latitude >= {query['latitude']['$gte']}")
        if '$lte' in query['latitude']:
            conditions.append(f"latitude <= {query['latitude']['$lte']}")
    
    return ' and '.join(conditions)


# Test cases
test_cases = [
    {
        "name": "query_condition_1 example",
        "json": '{"timestamp": {"$gte": 1730458317213, "$lte": 1730867875654}, "type_model": "IDX.2", "expert_collected": false}'
    },
    {
        "name": "query_condition_2 example",
        "json": '{"timestamp": {"$gte": 1728676386641, "$lte": 1733181529498}, "device_id": {"$in": ["DV181", "DV125", "DV282"]}, "tag_id": {"$in": ["68cd37349f89f5b6340db_183", "68cd37349f89f5b6340db_142"]}}'
    },
    {
        "name": "query_condition_3 example",
        "json": '{"timestamp": {"$gte": 1728699802843, "$lte": 1729708954843}, "sensor_lidar_type": {"$in": ["Pandar128", "AT256"], "$not_in": ["AT128"]}, "tag_id": {"$in": ["tag_1", "tag_2"], "contains_all": ["tag_3", "tag_4"]}, "longitude": {"$gte": 114.88, "$lte": 125.43}, "latitude": {"$gte": 25.85, "$lte": 34.90}}'
    }
]

print("=" * 80)
print("Testing JSONL Query to Milvus Expression Conversion")
print("=" * 80)
print()

for i, test_case in enumerate(test_cases, 1):
    print(f"Test Case {i}: {test_case['name']}")
    print("-" * 80)
    
    query = json.loads(test_case['json'])
    print(f"JSON Query:")
    print(json.dumps(query, indent=2))
    print()
    
    expression = convert_query_to_expression(query)
    print(f"Milvus Expression:")
    print(expression)
    print()
    print("=" * 80)
    print()

print("✅ All test cases completed!")

