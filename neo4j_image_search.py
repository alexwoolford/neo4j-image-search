#!/usr/bin/env python3
import numpy as np
import PIL.Image as Image
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
from neo4j import GraphDatabase

# get/instantiate classifier model
IMAGE_SHAPE = (224, 224)
classifier_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE + (3,))
])

# neo4j DB connection
uri = "neo4j://neo4j.woolford.io:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "V1ctoria"))

# dump all the nodes and relationships
with driver.session(database="photosearch") as session:
    session.run("MATCH(n) DETACH DELETE n")

# load the photos, as nodes, into the graph
pathlist = Path("photos").rglob('*.jpg')
for path in pathlist:
    path_in_str = str(path)
    image = Image.open(path).resize(IMAGE_SHAPE)
    image_array = np.array(image) / 255.0
    classification_array = classifier.predict(image_array[np.newaxis, ...])
    classification_list = [float(elem) for elem in list(classification_array[0])]

    with driver.session(database="photosearch") as session:
        session.run("MERGE (a:Photo {filename: $filename, classification_list: $classification_list})",
                    filename=path.name,
                    classification_list=classification_list)

# calculate the Cosine similarity scores between all the photos in the graph
with driver.session(database="photosearch") as session:
    session.run("MATCH (photoA:Photo) "
                "MATCH (photoB:Photo) "
                "WITH photoA, collect(photoB) AS photos "
                "UNWIND photos AS photoB "
                "WITH photoA, photoB, gds.alpha.similarity.cosine(photoA.classification_list, photoB.classification_list) AS similarity "
                "MERGE(:Photo {filename: photoA.filename})-[rel:IS_SIMILAR_TO {cosineSimilarity: similarity}]-(:Photo {filename: photoB.filename})")

# delete all the weak and self-relationships
with driver.session(database="photosearch") as session:
    session.run("MATCH(m:Photo)-[rel:IS_SIMILAR_TO]-(n:Photo) "
                "WHERE m.filename = n.filename "
                "OR rel.cosineSimilarity < 0.6 "
                "DELETE rel")

driver.close()
