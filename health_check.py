#!/usr/bin/env python3
"""
6-AI 프로젝트 서비스 상태 확인 스크립트
배포 전 필수 구성요소들이 모두 준비되었는지 체크
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import requests
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

class HealthChecker:
    def __init__(self):
        self.checks = []
        self.base_url = f"http://localhost:{os.getenv('PORT', '8000')}"
    
    def check_env_vars(self) -> Tuple[bool, str]:
        """필수 환경변수 확인"""
        required_vars = [
            'GEMINI_API_KEY',
            'JWT_SECRET_KEY'
        ]
        
        missing = []
        for var in required_vars:
            if not os.getenv(var):
                missing.append(var)
        
        if missing:
            return False, f"누락된 환경변수: {', '.join(missing)}"
        return True, "모든 필수 환경변수 설정됨"
    
    def check_directories(self) -> Tuple[bool, str]:
        """필수 디렉토리 존재 확인"""
        required_dirs = [
            'data/processed',
            'vector_db/chroma',
            'cache',
            'logs'
        ]
        
        missing = []
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                missing.append(dir_path)
        
        if missing:
            return False, f"누락된 디렉토리: {', '.join(missing)}"
        return True, "모든 필수 디렉토리 존재"
    
    def check_data_files(self) -> Tuple[bool, str]:
        """데이터 파일 존재 확인"""
        required_files = [
            'data/processed/processed_docs.jsonl',
            'vector_db/chroma/chroma.sqlite3'
        ]
        
        missing = []
        empty = []
        
        for file_path in required_files:
            path = Path(file_path)
            if not path.exists():
                missing.append(file_path)
            elif path.stat().st_size == 0:
                empty.append(file_path)
        
        if missing:
            return False, f"누락된 파일: {', '.join(missing)}"
        if empty:
            return False, f"빈 파일: {', '.join(empty)}"
        
        # 문서 수 확인
        try:
            with open('data/processed/processed_docs.jsonl', 'r', encoding='utf-8') as f:
                doc_count = sum(1 for _ in f)
            return True, f"데이터 준비 완료 ({doc_count}개 문서)"
        except Exception as e:
            return False, f"데이터 파일 읽기 실패: {e}"
    
    def check_dependencies(self) -> Tuple[bool, str]:
        """Python 패키지 의존성 확인"""
        required_packages = [
            'fastapi',
            'uvicorn',
            'langchain',
            'langchain_chroma',
            'langchain_huggingface',
            'transformers',
            'torch',
            'kiwipiepy'
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing.append(package)
        
        if missing:
            return False, f"누락된 패키지: {', '.join(missing)}"
        return True, "모든 의존성 패키지 설치됨"
    
    def check_api_server(self) -> Tuple[bool, str]:
        """API 서버 상태 확인"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                return True, f"API 서버 정상 동작 ({self.base_url})"
            else:
                return False, f"API 서버 응답 오류: {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, "API 서버 연결 불가 (서버가 실행 중인지 확인)"
        except Exception as e:
            return False, f"API 서버 확인 실패: {e}"
    
    def check_rag_service(self) -> Tuple[bool, str]:
        """RAG 서비스 기능 테스트"""
        try:
            # 간단한 테스트 토큰 생성 (실제 JWT는 아님)
            test_payload = {
                "query": "던파 테스트 질문",
                "jwtToken": "test_token_for_health_check"
            }
            
            response = requests.post(
                f"{self.base_url}/api/df/chat",
                json=test_payload,
                timeout=30
            )
            
            if response.status_code == 401:
                return True, "RAG 서비스 정상 (JWT 인증 활성화됨)"
            elif response.status_code == 200:
                data = response.json()
                if data.get('success') and data.get('answer'):
                    return True, "RAG 서비스 완전 정상"
                else:
                    return False, "RAG 서비스 응답 형식 오류"
            else:
                return False, f"RAG 서비스 오류: {response.status_code}"
                
        except Exception as e:
            return False, f"RAG 서비스 테스트 실패: {e}"
    
    def run_all_checks(self) -> Dict:
        """모든 체크 실행"""
        checks = [
            ("환경변수", self.check_env_vars),
            ("디렉토리", self.check_directories),
            ("데이터 파일", self.check_data_files),
            ("의존성", self.check_dependencies),
            ("API 서버", self.check_api_server),
            ("RAG 서비스", self.check_rag_service)
        ]
        
        results = {}
        all_passed = True
        
        for name, check_func in checks:
            try:
                passed, message = check_func()
                results[name] = {
                    'passed': passed,
                    'message': message
                }
                if not passed:
                    all_passed = False
            except Exception as e:
                results[name] = {
                    'passed': False,
                    'message': f"체크 실행 오류: {e}"
                }
                all_passed = False
        
        results['overall'] = all_passed
        return results
    
    def print_results(self, results: Dict):
        """결과 출력"""
        print("🔍 6-AI 서비스 상태 체크 결과")
        print("=" * 50)
        
        for name, result in results.items():
            if name == 'overall':
                continue
                
            status = "✅" if result['passed'] else "❌"
            print(f"{status} {name:12} | {result['message']}")
        
        print("=" * 50)
        
        if results['overall']:
            print("🎉 모든 체크 통과! 서비스 준비 완료")
            print("🚀 배포 가능 상태입니다")
        else:
            print("⚠️  일부 체크 실패")
            print("📋 위의 문제들을 해결한 후 다시 확인해주세요")
        
        print()
        return results['overall']


def main():
    """메인 함수"""
    print("🔍 6-AI 서비스 상태 확인 중...\n")
    
    checker = HealthChecker()
    results = checker.run_all_checks()
    success = checker.print_results(results)
    
    # 추가 정보 출력
    if success:
        print("📋 추가 정보:")
        print(f"   🌐 API 문서: {checker.base_url}/docs")
        print(f"   📊 로그 위치: logs/")
        print(f"   ⚙️  설정 파일: .env")
        
        # 데이터 통계
        try:
            with open('data/processed/processed_docs.jsonl', 'r', encoding='utf-8') as f:
                doc_count = sum(1 for _ in f)
            print(f"   📚 문서 수: {doc_count:,}개")
        except:
            pass
    else:
        print("🛠️  문제 해결 방법:")
        print("   1. pipeline.py 실행으로 데이터 준비")
        print("   2. .env 파일에서 API 키 설정 확인")
        print("   3. pip install -r requirements.txt 재실행")
        print("   4. python -m api.main으로 서버 시작")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
