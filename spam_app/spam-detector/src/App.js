import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import './App.css';

function App() {
  const [input, setInput] = useState('');
  const [result, setResult] = useState('');
  const [model, setModel] = useState(null);

  useEffect(() => {
    async function loadModel() {
      try {
        const loadedModel = await tf.loadLayersModel('Rabble/model_LSTM.py');
        setModel(loadedModel);
      } catch (error) {
        console.error('모델 로딩 중 오류 발생:', error);
      }
    }
    loadModel();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (model) {
      try {
        // 여기에 실제 전처리 및 예측 로직을 구현하세요
        // 예시:
        const processedInput = preprocessText(input);
        const prediction = await model.predict(processedInput);
        setResult(prediction.dataSync()[0] > 0.5 ? '스팸' : '정상');
      } catch (error) {
        console.error('예측 중 오류 발생:', error);
        setResult('예측 실패');
      }
    } else {
      setResult('모델이 아직 로드되지 않았습니다.');
    }
  };

  // 전처리 함수 예시 (실제 구현 필요)
  const preprocessText = (text) => {
    // 여기에 실제 전처리 로직을 구현하세요
    return tf.tensor2d([text.split(' ').map(word => 1)]); // 예시일 뿐입니다
  };

  return (
    <div className="App">
      <div class="header">
        <img src="/spam_guardian-removebg-preview.png"/>
        <h1>Spam Guardian</h1>
      </div>
      <form onSubmit={handleSubmit}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="텍스트를 입력하세요"
        />
        <button type="submit">스팸 검사</button>
      </form>
      {result && <p>결과: {result}</p>}
    </div>
  );
}

export default App;