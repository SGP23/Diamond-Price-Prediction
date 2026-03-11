import logging
from src.DiamondPricePrediction.logger import logging as _  # noqa: F401 — initializes logging config
from src.DiamondPricePrediction.pipelines.Prediction_Pipeline import CustomData, PredictPipeline

from flask import Flask, request, render_template

logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")

    else:
        logger.info("Received prediction request")
        data = CustomData(
            carat=float(request.form.get('carat')),
            depth=float(request.form.get('depth')),
            table=float(request.form.get('table')),
            x=float(request.form.get('x')),
            y=float(request.form.get('y')),
            z=float(request.form.get('z')),
            cut=request.form.get('cut'),
            color=request.form.get('color'),
            clarity=request.form.get('clarity')
        )
        final_data = data.get_data_as_dataframe()
        logger.info(f"Input features: carat={data.carat}, cut={data.cut}, color={data.color}, clarity={data.clarity}")

        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_data)
        result_usd = round(pred[0], 2)

        # Convert USD to INR (fixed rate: 1 USD = 92 INR)
        USD_TO_INR = 92
        result_inr = round(result_usd * USD_TO_INR, 2)

        logger.info(f"Predicted price: ${result_usd} USD / ₹{result_inr} INR")
        return render_template("result.html", final_result=result_inr, price_usd=result_usd)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
