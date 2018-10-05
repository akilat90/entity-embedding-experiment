from keras.models import model_from_json
from sklearn.externals import joblib

import pandas as pd
import embed_helpers as eh

import flask
from StringIO import StringIO
import tensorflow as tf

app = flask.Flask(__name__)
model = None


def load_models():

    global model, scaler, graph

    # model and weight paths
    model_path, weights_path = 'models/nn_basic_cv.json', 'models/nn_basic_cv_weights.h5'

    # load json and create model
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # load weights into new model
    model.load_weights(weights_path)

    # compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['acc'])

    # load scaler
    scaler_path = 'models/scaler.pickle'
    scaler = joblib.load(scaler_path)

    graph = tf.get_default_graph()


def make_me_acceptable(sample):

    # normalize the numeric columns
    sample_normalized = eh.normalize_numeric_columns(sample,
                                                     columns=[c for c in sample.columns if (
                                                         not c.endswith('_cat'))],
                                                     scaler=scaler,
                                                     is_production=True
                                                     )
    cat_names = ['workclass_cat',
                 'education_cat',
                 'marital.status_cat',
                 'occupation_cat',
                 'relationship_cat',
                 'race_cat',
                 'sex_cat',
                 'native.country_cat']

    # make the input list
    input_list = eh.make_input_list(sample_normalized, cat_names)

    return input_list


@app.route("/predict", methods=["POST"])
def predict():

    # to be sent to the client
    data = {"success": False}

    if flask.request.method == "POST":

        try:
            # read file from request
            file = flask.request.files['file'].read()

            # StringIO so that pandas can read it
            file = StringIO(file)
            sample = pd.read_csv(file)

            # prepare the input list that the loaded model accepts
            input_list = make_me_acceptable(sample)

            # load model ()
            with graph.as_default():
                prediction = model.predict(input_list)

            # do the prediction
            prediction = prediction.tolist()

            # to be sent to client
            data["prediction"] = prediction

            data["success"] = True

        except Exception as e:
            # TODO: handle exceptions specifically - gotta read a bit
            pass

    # return a json serialized version of data
    return flask.json.dumps(data)


if __name__ == "__main__":
    print(("* Loading models and starting server..."))
    load_models()
    app.run()
