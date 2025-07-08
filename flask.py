from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/receive_string', methods=['POST'])
def receive_string():
    data = request.get_json()
    text = data.get('text', '')

    print(f"[RECEIVED STRING] {text}")
    
    # Kamu bisa proses atau simpan string di sini
    return jsonify({'message': f'Received: {text}'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
