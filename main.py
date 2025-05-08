from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import base64
import os
import logging
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig

# ロギング設定
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# VertexAIの初期化
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("REGION", os.getenv("LOCATION", "us-central1"))

@app.before_request
def initialize_vertexai():
    if PROJECT_ID is None:
        app.logger.error("PROJECT_ID環境変数が設定されていません")
    else:
        try:
            vertexai.init(project=PROJECT_ID, location=LOCATION)
            app.logger.info(f"VertexAI初期化成功: プロジェクト={PROJECT_ID}, リージョン={LOCATION}")
        except Exception as e:
            app.logger.error(f"VertexAI初期化エラー: {str(e)}")

@app.route('/generate-image', methods=['POST'])
def generate_image():
    try:
        app.logger.info("画像生成リクエストを受信しました")
        
        # PROJECT_IDの確認
        if PROJECT_ID is None:
            return jsonify({"error": "PROJECT_ID環境変数が設定されていません"}), 500
            
        # リクエストからプロンプトを取得
        data = request.get_json()
        if not data or 'prompt' not in data:
            app.logger.error("無効なリクエスト: プロンプトがありません")
            return jsonify({"error": "プロンプトが必要です"}), 400
        
        prompt = data['prompt']
        app.logger.info(f"プロンプト: {prompt}")
        
        # Geminiモデルで画像生成
        try:
            model = GenerativeModel(model_name="gemini-2.0-flash-preview-image-generation")
            app.logger.info("Geminiモデル初期化成功")
            
            generation_config = GenerationConfig(
                response_mime_type="image/png"
            )
            
            response = model.generate_content(
                contents=prompt,
                generation_config=generation_config
            )
            app.logger.info("Gemini API呼び出し成功")
        except Exception as api_error:
            app.logger.error(f"Gemini API呼び出しエラー: {str(api_error)}")
            return jsonify({"error": f"Gemini API呼び出しエラー: {str(api_error)}"}), 500
        
        # レスポンスを処理
        result = {"success": True}
        
        if hasattr(response, 'text'):
            result["text"] = response.text
        
        try:
            # 画像を取得
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    # 画像データをBase64エンコード
                    image_data = part.inline_data.data
                    image = Image.open(BytesIO(image_data))
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    result["image"] = img_str
                    app.logger.info("画像処理成功")
        except Exception as img_error:
            app.logger.error(f"画像処理エラー: {str(img_error)}")
            return jsonify({"error": f"画像処理エラー: {str(img_error)}"}), 500
        
        return jsonify(result)
    
    except Exception as e:
        app.logger.error(f"サーバーエラー: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "project": PROJECT_ID, "location": LOCATION}), 200

@app.route('/', methods=['GET'])
def root():
    return 'Gemini Image API is running!'

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
