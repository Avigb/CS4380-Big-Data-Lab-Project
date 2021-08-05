from kafka import KafkaProducer
from google.cloud import storage

#kafka details
IP_ = "10.128.0.48:9092"
topic_name = 'PROJECT'
producer = KafkaProducer(bootstrap_servers = [IP_])

#download data from gcs
client = storage.Client()
bucket = client.get_bucket("bdl2021_final_project")
blobs_all = list(bucket.list_blobs(prefix="nyc_tickets_train.csv/"))

#print(blobs_all[2:])

i = 0
for blob in blobs_all[2:]:                  #ignore first 2 irelevant files

    #download data from GCS
    content = blob.download_as_string()
    content = content.decode('utf-8')

    #split and iterate over aeach line
    lines = content.split('\n')
    for line in lines[1:]:                  #ignore title line
        if line != '':
            line = bytes(line,'utf-8')
            producer.send(topic_name,line)
            producer.flush()

    print("{} lines from part no: {} written to Kafka".format(len(lines),i))
    i += 1
