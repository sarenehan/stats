# Upload generate_prob_app_ownership_model.py to s3:
aws s3 cp final_project.py s3://stat-538/final_project.py
aws s3 cp aws_emr_tools/install_python_packages.sh s3://stat-538/install_python_packages.sh
# Creating the cluster:
aws emr create-cluster \
--applications Name=Hadoop Name=Hive Name=Spark Name=Ganglia \
--ec2-attributes \
    '{"KeyName":"stats", "InstanceProfile":"EMR_EC2_DefaultRole", "SubnetId": "subnet-f84456a1"}'  \
--service-role EMR_DefaultRole \
--enable-debugging \
--release-label emr-5.9.0 \
--log-uri 's3n://stat-538/logs/' \
--name 'deep learning' \
--bootstrap-actions file:///Users/stewart/projects/stats/538/aws_emr_tools/bootstrap.json \
--instance-groups file:///Users/stewart/projects/stats/538/aws_emr_tools/instance_groups_configuration.json \
--region us-west-2 \
--configurations file:///Users/stewart/projects/stats/538/aws_emr_tools/emr_configuration.json \
--steps Type=spark,Name=DeepLearning,ActionOnFailure=CONTINUE,Args=[s3://stat-538/final_project.py] \
--auto-terminate