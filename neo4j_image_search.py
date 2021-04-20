#!/usr/bin/env python3
import numpy as np
import PIL.Image as Image
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
from neo4j import GraphDatabase

# get/instantiate classifier model
IMAGE_SHAPE = (224, 224)
classifier_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/3"
classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE + (3,), output_shape=[1001])
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
    session.run("MATCH (p:Photo) "
                "WITH {item:id(p), weights: p.classification_list} AS itemData "
                "WITH collect(itemData) AS data "
                "CALL gds.alpha.similarity.cosine.write({ "
                "  data: data, "
                "  skipValue: null, "
                "  similarityCutoff:0.6, "
                "  writeRelationshipType:'IS_SIMILAR_TO' "
                "}) "
                "YIELD nodes, similarityPairs, min, max, mean, stdDev "
                "RETURN nodes, similarityPairs, min, max, mean, stdDev")

driver.close()
