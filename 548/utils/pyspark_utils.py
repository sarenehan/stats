from contextlib import contextmanager
from pyspark import SparkContext, SparkConf


@contextmanager
def get_spark_context(
    app_name="localPyspark",
    n_cores="*",
    driver_memory="10G",
    executor_memory="10G",
    maxResultSize="5G"
):
    conf = SparkConf().setAppName("App")
    conf = (conf.setMaster('local[{}]'.format(n_cores))
            .set('spark.executor.memory', executor_memory)
            .set('spark.driver.memory', driver_memory)
            .set('spark.driver.maxResultSize', maxResultSize))
    sc = SparkContext(conf=conf)
    try:
        yield sc
    finally:
        sc.stop()


def transform_list_parallel(
        list_to_transform, transform_func, *transform_args):
    with get_spark_context() as sc:
        args_to_pass = []
        for arg in transform_args:
            new_arg = sc.broadcast(arg)
            args_to_pass.append(new_arg.value)
        # broadcast_args = [sc.broadcast(arg).value for arg in transform_args]
        result = sc.parallelize(list_to_transform, len(list_to_transform)).map(
            lambda element: transform_func(element, *args_to_pass)
        ).collect()
    return result
