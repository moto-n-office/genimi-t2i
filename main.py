from flask import Flask, request, jsonify, send_file
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64
import os

app = Flask(__name__)

# Google APIキーを環境変数から取得
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

@app.route('/generate-image', methods=['POST'])
def generate_image():
    try:
        # リクエストからプロンプトを取得
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"error": "プロンプトが必要です"}), 400
        
        prompt = data['prompt']
        
        # Geminiモデルで画像生成
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        
        # レスポンスを処理
        result = {"success": True}
        
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                result["text"] = part.text
            elif part.inline_data is not None:
                # 画像データをBase64エンコード
                image = Image.open(BytesIO(part.inline_data.data))
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                result["image"] = img_str
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
