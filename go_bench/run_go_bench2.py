import os
import time
import logging
import sys
def setup_logging():
    # Create log directory if it doesn't exist
    log_dir = "/tmp"
    log_file = os.path.join(log_dir, "gobench_query.log")
    
    # Configure logging format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging initialized. Log file: %s", log_file)


uri = "https://in01-d1a7exxxx92ad5.aws-us-west-2.vectordb-uat3.zillizcloud.com:19535"
user = "mock-tke-user"
password = "mock-tke-password"

expr_list = [
            'int64_1 == {0}', 
            'int64_1 <= {0}', 
            '0 <= float_1 <= {0}',
            'int64_1 in X', 
            'varchar_1 in X',
            'array_int64[1] in X',
            'array_varchar[0] in X',
            'json_1["key_0"]["key"] == {0}',
            'varchar_1 like "%{0}"',
            'array_contains_any(array_varchar, [\"{0}0\", \"{0}1\", \"{0}2\", \"{0}3\", \"{0}4\", \"{0}5\", \"{0}6\", \"{0}7\", \"{0}8\", \"{0}9\"])',
            'array_contains(array_int64, {0})',
            'mixed_conditions'
             ]

range_list = [
    [0, 2200000],
    [0, 2200000],
    [1, 1000.0],
    [1000, [0, 2200000], 'int64_1', 'int64'],
    [1000, [1000000000, 1000010000], 'varchar_1', 'varchar'],
    [100, [0, 1000], 'array_int64[1]', 'int64'],
    [100, [0, 1000], 'array_varchar[0]', 'varchar'],
    [0, 100],
    [0, 100],
    [0, 100],
    [0, 1000],
    ['int64_1 > 100 || float_1 > 100.0 || ', 10, [0, 100], 'json_1["key_0"]["key"]', 'int64', 'array_contains_any(array_varchar, [\"{0}0\", \"{0}1\", \"{0}2\", \"{0}3\", \"{0}4\", \"{0}5\", \"{0}6\", \"{0}7\", \"{0}8\", \"{0}9\"])', [0, 100]]
]

output_fields_list=[
                    ["count(*)"],
                    ["int64_1"],
                    ["varchar_1"],
                    ["varchar_2"],
                    ["array_int64"],
                    ["array_varchar"],
                    ["json_1"],
                    ["varchar_1", "array_int64", "json_1"],
                    ["varchar_1", "array_int64"]
                    ]


# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

for i in range(len(expr_list)):
    expr = expr_list[i]
    use_in = True if ' in X' in str(expr) else False
    mixed_conditions = True if 'mixed_conditions' in expr else False
    range = range_list[i]
    for output_fields in output_fields_list:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        logging.info(f"[{timestamp}] expr: {expr}, output_fields: {output_fields}")
        with open('config.yaml', 'r') as f:
            lines = f.readlines()
        # Update configuration file
        for i, line in enumerate(lines):
            # Use regex to match custom_expr line regardless of specific value
            if 'expr:' in line and 'custom_expr:' not in line:
                if mixed_conditions:
                    lines[i] = f"      expr: {range[0]}\n"
                else:
                    lines[i] = f"      expr: null\n"
            if 'custom_expr:' in line:
                if use_in:
                    lines[i] = f"      custom_expr: null\n"
                else:
                    lines[i] = f"      custom_expr: {expr}\n"
                if mixed_conditions:
                    lines[i] = f"      custom_expr: {range[5]}\n"
            if 'random_data:' in line:
                if use_in:
                    lines[i] = f"      random_data: true\n"
                else:
                    lines[i] = f"      random_data: false\n"
                if mixed_conditions:
                    lines[i] = f"      random_data: true\n"
            if 'random_count:' in line:
                if use_in:
                    lines[i] = f"      random_count: {range[0]}\n"
                else:
                    pass
                if mixed_conditions:
                    lines[i] = f"      random_count: {range[1]}\n"
            if 'random_range:' in line:
                if use_in:
                    lines[i] = f"      random_range: {range[1]}\n"
                else:
                    pass
                if mixed_conditions:
                    lines[i] = f"      random_range: {range[2]}\n"
            if 'field_name:' in line:
                if use_in:
                    lines[i] = f"      field_name: {range[2]}\n"
                else:
                    pass
                if mixed_conditions:
                    lines[i] = f"      field_name: {range[3]}\n"
            if 'field_type:' in line:
                if use_in:
                    lines[i] = f"      field_type: {range[3]}\n"
                else:
                    pass
                if mixed_conditions:
                    lines[i] = f"      field_type: {range[4]}\n"
            if 'custom_range:' in line:
                if not use_in:
                    lines[i] = f"      custom_range: {range}\n"
                else:
                    pass
                if mixed_conditions:
                    lines[i] = f"      custom_range: {range[6]}\n"
            # Use regex to match output_fields line regardless of specific value    
            if 'output_fields:' in line:
                output_fields_str = str(output_fields).replace("'", '"')
                lines[i] = f'      output_fields: {output_fields_str}\n'
            if 'limit:' in line:
                if output_fields == ["count(*)"]:
                    lines[i] = "      limit: null\n"
                else:
                    lines[i] = "      limit: 1000\n"
        with open('config.yaml', 'w') as f:
            f.writelines(lines)
        
        time.sleep(180)
        # Generate unique log filename with timestamp
        log_file = os.path.join('logs', f"bench_{timestamp}.log")
        # Execute go bench and redirect output to log file
        os.system(f'./benchmark parallel -c config.yaml -u {uri} -n {user} -p {password} > {log_file} 2>&1')

logging.info("Done")