import math

def generate_square_polygon(center_point, area_sq_km):
    """
    生成以给定点为中心、指定面积的正方形区域的WKT POLYGON
    
    参数:
    center_point: 中心点，可以是字符串"经度 纬度"或元组(经度, 纬度)
    area_sq_km: 正方形区域的面积（平方公里）
    
    返回:
    WKT格式的POLYGON字符串
    """
    # 解析中心点
    if isinstance(center_point, str):
        lon, lat = map(float, center_point.split())
    else:
        lon, lat = center_point
    
    # 地球半径（公里）
    earth_radius = 6371.0
    
    # 计算纬度每度的距离（公里）
    km_per_degree_lat = 111.195
    
    # 计算当前纬度下经度每度的距离（公里）
    km_per_degree_lon = math.cos(math.radians(lat)) * 111.195
    
    # 计算正方形的边长（公里）
    side_length_km = math.sqrt(area_sq_km)
    
    # 计算半边长（从中心到边的距离）
    half_side = side_length_km / 2
    
    # 计算纬度和经度的偏移量
    lat_offset = half_side / km_per_degree_lat
    lon_offset = half_side / km_per_degree_lon
    
    # 计算四个角点
    sw_lon = lon - lon_offset  # 西南角经度
    sw_lat = lat - lat_offset  # 西南角纬度
    
    se_lon = lon + lon_offset  # 东南角经度
    se_lat = lat - lat_offset  # 东南角纬度
    
    ne_lon = lon + lon_offset  # 东北角经度
    ne_lat = lat + lat_offset  # 东北角纬度
    
    nw_lon = lon - lon_offset  # 西北角经度
    nw_lat = lat + lat_offset  # 西北角纬度
    
    # 生成WKT格式的POLYGON
    wkt_polygon = f"POLYGON(({sw_lon} {sw_lat}, {se_lon} {se_lat}, {ne_lon} {ne_lat}, {nw_lon} {nw_lat}, {sw_lon} {sw_lat}))"
    
    return wkt_polygon

# 示例用法
if __name__ == "__main__":
    # 示例点列表
    points = [
        'POINT (-73.982102 40.73629)',
        'POINT (-74.002587 40.739748)',
        'POINT (-73.974267 40.790955)',
        'POINT (-74.00158 40.719382)',
        'POINT (-73.98404999999998 40.743544)',
        'POINT (-73.96969 40.749244)'
    ]
    
    # 提取坐标部分
    coordinates = []
    for point in points:
        # 提取括号内的坐标部分
        coords = point[point.find('(')+1:point.find(')')]
        coordinates.append(coords)
    
    # 为每个点生成2平方公里的正方形
    area = 1  # 平方公里
    for i, coord in enumerate(coordinates):
        polygon = generate_square_polygon(coord, area)
        print(f"Point {i+1}: {points[i]}")
        print(f"Polygon: {polygon}\n")