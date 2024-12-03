# server.py

from flask import Flask, Response, stream_with_context, request, jsonify
from flask_cors import CORS
import os
from ServerTee import ServerTee
from thread_handler import ThreadHandler
from WorkFlow import run_workflow_as_server
from FileTransmit import file_transmit_bp  # Import the Blueprint

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

app.register_blueprint(file_transmit_bp)  # Register the Blueprint

server_tee = ServerTee("server.log")
thread_handler = ThreadHandler.get_instance()

def server_func(params):
    try:
        run_workflow_as_server(params)
    except Exception as e:
        print(str(e))
        raise

@app.route('/run', methods=['POST'])
def run_script():
    if thread_handler.is_running():
        return "Another instance is already running", 409

    # Grab the body parameters
    try:
        if request.is_json:  # Check if the request is JSON
            body_params = request.get_json()
        else:  # Fallback for form-encoded data
            body_params = request.form.to_dict()
        print("Received body parameters:", body_params)
    except Exception as e:
        return jsonify({"error": "Invalid request body", "details": str(e)}), 400
    print('body_params ', body_params)
    def generate():
        try:
            thread_handler.start_thread(target=server_func, params=body_params)
            yield from server_tee.stream_to_frontend()
        except Exception as e:
            print(str(e))
            yield "Error occurred: " + str(e)
        finally:
            if not thread_handler.is_running():
                yield "finished"

    return Response(stream_with_context(generate()), content_type='text/plain; charset=utf-8')

@app.route('/stop', methods=['POST'])
def stop_script():
    if thread_handler.is_running():
        thread_handler.force_reset()
        return "Script stopped", 200
    else:
        return "No script is running", 400

@app.route('/status', methods=['GET'])
def check_status():
    running = thread_handler.is_running()
    return jsonify({"running": running}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
