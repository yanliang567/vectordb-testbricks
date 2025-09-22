import os
import time
import logging
import sys
import random
import datetime

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


uri = "https://in01-9028520cb1d63cf.ali-cn-hangzhou.cloud-uat.zilliz.cn:19530"
user = "yanliang"
password = "Milvus123"

def generate_random_expression(expr_key):
    """Generate random query expression"""

    device_id_keywords = [
        'SENSOR_A123', 'SENSOR_A233', 'SENSOR_A108', 'SENSOR_A172', 'CAM_B112', 
        'CAM_B177', 'DV348', 'DV378', 'DV081', 'DV349']
    
    polygon_keywords = [
        "'POLYGON((-73.991957 40.721567, -73.982102 40.73629, -74.002587 40.739748, -73.974267 40.790955, -73.991957 40.721567))'"  # NYC area
        # "'POLYGON((-74.1 40.6, -73.8 40.6, -73.8 40.9, -74.1 40.9, -74.1 40.6))'",  # Larger NYC area
        # "'POLYGON((-74.05 40.75, -73.95 40.75, -73.95 40.85, -74.0""" 40.85, -74.05 40.75))'"  # Central NYC
    ]

    if expr_key is None:
        return None

    if expr_key.lower() == "equal":
        device_id_keyword = random.choice(device_id_keywords)
        return f'device_id == "{device_id_keyword}"'    #  'device_id == \"SENSOR_A123\"'

    elif expr_key.lower() == "equal_and_expert_collected":
        device_id_keyword = random.choice(device_id_keywords)
        return f'device_id == "{device_id_keyword}" and expert_collected == True'

    elif expr_key.lower() == "equal_and_timestamp_week":
        device_id_keyword = random.choice(device_id_keywords)
        # Generate a random 7-day window between 2025-01-01 and 2025-08-30
        import datetime

        # Define the start and end date range
        start_date = datetime.datetime(2025, 1, 1)
        end_date = datetime.datetime(2025, 8, 23)  # So that start + 7 days <= 2025-08-30

        # Calculate the total days between start and end
        total_days = (end_date - start_date).days

        # Randomly pick a start day offset
        random_offset = random.randint(0, total_days)
        window_start_date = start_date + datetime.timedelta(days=random_offset)
        window_end_date = window_start_date + datetime.timedelta(days=6)  # 7 days window

        # Convert to timestamp (assume UTC)
        left_timestamp_keyword = int(window_start_date.replace(tzinfo=datetime.timezone.utc).timestamp())
        right_timestamp_keyword = int(window_end_date.replace(tzinfo=datetime.timezone.utc).timestamp())
        return f'device_id == "{device_id_keyword}" and timestamp >= {left_timestamp_keyword} and timestamp <= {right_timestamp_keyword}'

    elif expr_key.lower() == "equal_and_timestamp_month":
        device_id_keyword = random.choice(device_id_keywords)
        # Generate a random 30-day window between 2025-01-01 and 2025-08-30
        import datetime

        # Define the start and end date range
        start_date = datetime.datetime(2025, 1, 1)
        end_date = datetime.datetime(2025, 8, 1)  # So that start + 30 days <= 2025-08-30

        # Calculate the total days between start and end
        total_days = (end_date - start_date).days

        # Randomly pick a start day offset
        random_offset = random.randint(0, total_days)
        window_start_date = start_date + datetime.timedelta(days=random_offset)
        window_end_date = window_start_date + datetime.timedelta(days=29)  # 30 days window

        # Convert to timestamp (assume UTC)
        left_timestamp_keyword = int(window_start_date.replace(tzinfo=datetime.timezone.utc).timestamp())
        right_timestamp_keyword = int(window_end_date.replace(tzinfo=datetime.timezone.utc).timestamp())

        return f'device_id == "{device_id_keyword}" and timestamp >= {left_timestamp_keyword} and timestamp <= {right_timestamp_keyword}'

    elif expr_key.lower() == "geo_contains":
        polygon = random.choice(polygon_keywords)
        polygon = "'POLYGON((-73.991957 40.721567, -73.982102 40.73629, -74.002587 40.739748, -73.974267 40.790955, -73.991957 40.721567))'"
        return f'ST_CONTAINS(location, {polygon})'
        
    elif expr_key.lower() == "sensor_contains":
        keywords = [
            'Thor_Trucks', 'WeRide_Robobus', 'Delphi_ESR', 'Aptiv_SRR4', 'AEye_iDAR', 'DiDi_Gemini', 'ADAS_Eyes', 
            'Embark_Guardian', 'Hella_24GHz', 'ST_VL53L1X', 'TuSimple_AFV', 'Locomation_AutonomousRelay', 'Voyage_Telessport', 
            'Livox_Horizon', 'Infineon_BGT24', 'Aurora_FirstLight', 'Ibeo_LUX', 'Ouster_OS1_64', 'Delphi_ESR']
        keyword = random.choice(keywords)
        return f'ARRAY_CONTAINS(sensor_lidar_type, "{keyword}")'

    elif expr_key.lower() == "device_id_in":
        device_ids = random.sample(device_id_keywords, 3)
        return f'device_id in {device_ids}'
    elif expr_key.lower() == "sensor_json_contains":
        keywords = [
            'Thor_Trucks', 'WeRide_Robobus', 'Delphi_ESR', 'Aptiv_SRR4', 'AEye_iDAR', 'DiDi_Gemini', 'ADAS_Eyes', 
            'Embark_Guardian', 'Hella_24GHz', 'ST_VL53L1X', 'TuSimple_AFV', 'Locomation_AutonomousRelay', 'Voyage_Telessport', 
            'Livox_Horizon', 'Infineon_BGT24', 'Aurora_FirstLight', 'Ibeo_LUX', 'Ouster_OS1_64']   # Delphi_ESR
        keyword = random.sample(keywords, 3)
        return f'JSON_CONTAINS_ALL(sensor_lidar_type, {keyword}) AND NOT JSON_CONTAINS(sensor_lidar_type, "Delphi_ESR")'
    else:
        return None


# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

expr_keys = [
            ('equal', 10), ('equal', 20), 
            ('equal_and_expert_collected', 10), ('equal_and_expert_collected', 20), 
            ('equal_and_timestamp_week', 20), ('equal_and_timestamp_week', 30),
            ('equal_and_timestamp_month', 20), ('equal_and_timestamp_month', 30),
            ('geo_contains', 10), ('geo_contains', 20),
            ('sensor_contains', 10), ('sensor_contains', 20),
            ('device_id_in', 10), ('device_id_in', 20),
            ('sensor_json_contains', 10), ('sensor_json_contains', 20)
            ]

# expr_keys = ['equal']

for expr_key, concurrent_number in expr_keys:
    expr = generate_random_expression(expr_key)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join('/root/test/logs', f"bench_{expr_key}_{concurrent_number}_{timestamp}.log")
    with open('config.yaml', 'r') as f:
        lines = f.readlines()
    # Update configuration file
    for i, line in enumerate(lines):
        # Use regex to match custom_expr line regardless of specific value
        if 'expr:' in line and 'custom_expr:' not in line:
            lines[i] = f"      expr: {expr}\n"
        if 'concurrent_number:' in line:
            lines[i] = f"  concurrent_number: {concurrent_number}\n"
    with open('config.yaml', 'w') as f:
        f.writelines(lines)
    
    time.sleep(120)

    # Execute go bench and redirect output to log file
    os.system(f'./benchmark parallel -c config.yaml -u {uri} -n {user} -p {password} > {log_file} 2>&1')
    # os.system(f'./benchmark parallel -c config.yaml -u {uri} -n {user} -p {password}')

logging.info("Done")