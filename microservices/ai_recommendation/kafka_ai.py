from kafka import KafkaConsumer
import json
from gemini import get_personal_recommendations
from recommendationEngine import TemporalRecommendationEngine

def recommend_kafka(topic_name):
    print("STARTING!", flush=True)
    engine = TemporalRecommendationEngine()
    engine.initialize()

    consumer = KafkaConsumer(
        bootstrap_servers=['kafka:9092'],
        group_id='my-consumer-group',
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        value_deserializer=lambda x: x.decode('utf-8') if x else None
    )
    consumer.subscribe([topic_name])
    print(consumer.topics())
    print("SUBSCRIBED", flush=True)
    # 
    print(f"Listening to topic: {topic_name}", flush=True)

    try:
        while True:
            message_pack = consumer.poll(timeout_ms=1000)
            
            if not message_pack:
                print("No messages received, continuing...", flush=True)
                continue
                
            for topic_partition, messages in message_pack.items():
                for message in messages:
                    print(f"Received: {message.value}", flush=True)
                    print(f"Topic: {message.topic}, Partition: {message.partition}", flush=True)
                    mes = json.loads(message.value)
                    user_id = mes['user_id']
                    print(engine.get_temporal_recommendations(user_id, num_recommendations=10))
                
    except KeyboardInterrupt:
        print("Consumer interrupted", flush=True)
    finally:
        consumer.close()
    # 
    # try:
    #     for message in consumer:
    #         print("TRIED")
    #         print(f"Received: {message.value}")
    #         
    #         # Process your message here
    #         try:
    #             data = json.loads(message.value)
    #             print(f"JSON data: {data}")
    #         except json.JSONDecodeError:
    #             print("Non-JSON message received")
    #             
    # except Exception as e:
    #     print("\nStopping consumer...", e)
    # finally:
    #     consumer.close()
# print("beginning")
# # consumer = RecommendationKafka()
# simple_kafka_consumer('user_interaction_ai')
# # consumer.start_listening()
if __name__ == "__main__":
    recommend_kafka('user_interaction_ai')
