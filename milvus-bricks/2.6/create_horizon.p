{
        'collection_name': 'horizon_test_collection',
        'auto_id': True,
        'num_shards': 1,
        'description': '',
        'fields': [{
                'field_id': 100,
                'name': 'id',        # mock
                'description': '',
                'type': < DataType.VARCHAR: 21 > ,
                'params': {
                        'max_length': 64
                },
                'auto_id': True,
                'is_primary': True
        }, {
                'field_id': 101,
                'name': 'feature',     # 地平线提供
                'description': '',
                'type': < DataType.FLOAT_VECTOR: 101 > ,
                'params': {
                        'dim': 768
                }
        }, {
                'field_id': 102,
                'name': 'timestamp',    # mock样例1751961530077 过去16个月之内，output
                'description': '',     
                'type': < DataType.INT64: 5 > ,
                'params': {}
        }, {
                'field_id': 103,
                'name': 'url',            # mock，output
                'description': '',
                'type': < DataType.VARCHAR: 21 > ,
                'params': {
                        'max_length': 1024
                }
        }, {
                'field_id': 104,
                'name': 'device_id',    # mock样例："DV345"，分布基数1000个，长度10-20，output
                'description': '',
                'type': < DataType.VARCHAR: 21 > ,
                'params': {
                        'max_length': 32
                }
        }, {
                'field_id': 105,
                'name': 'longitude',    # from NYC-Taxi， 改造成location字段，output
                'description': '',
                'type': < DataType.FLOAT: 10 > ,
                'params': {}
        }, {
                'field_id': 106,
                'name': 'latitude',    # from NYC-Taxi，改造成location字段，output
                'description': '',
                'type': < DataType.FLOAT: 10 > ,
                'params': {}
        }],
        'aliases': [],
        'collection_id': 45733527117584,
        'consistency_level': 2,
        'properties': {},
        'num_partitions': 16,
        'enable_dynamic_field': True
}

