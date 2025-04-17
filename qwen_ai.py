import requests
import base64
import json

class QwenAI:
    """
    使用Qwen API進行生成式AI功能的類，支持免費API選項
    """
    def __init__(self, api_key=None, api_url=None):
        self.api_key = api_key
        self.api_url = api_url or "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.model = "qwen-max"  # 使用Qwen-Max模型
        
        # 使用免費API
        self.use_free_api = api_url is None and (api_key is None or api_key.strip() == "")
        self.free_api_url = "https://api.qwen-2.com/v1/chat/completions"
    
    def is_available(self):
        """檢查API是否可用"""
        return True  # 始終可用，因為我們有免費API選項
    
    def analyze_comparison_results(self, comparison_results):
        """分析比對結果並提供見解"""
        try:
            # 準備比對結果摘要
            summary = {
                "total_paragraphs": len(comparison_results),
                "matched_paragraphs": sum(1 for r in comparison_results if r["similarity"] >= 0.8),
                "unmatched_paragraphs": sum(1 for r in comparison_results if r["similarity"] < 0.8),
                "examples": comparison_results[:3]  # 提供前3個例子
            }
            
            summary_text = f"""
比對結果摘要:
- 總段落數: {summary['total_paragraphs']}
- 匹配段落數: {summary['matched_paragraphs']}
- 未匹配段落數: {summary['unmatched_paragraphs']}

例子:
"""
            for i, example in enumerate(summary["examples"]):
                summary_text += f"""
例子 {i+1}:
原始文本: {example['original_text'][:100]}...
美編後文本: {example['edited_text'][:100]}...
相似度: {example['similarity']}
"""
            
            # 使用API進行分析
            if self.use_free_api:
                return self._analyze_with_free_api(summary_text)
            else:
                return self._analyze_with_official_api(summary_text)
        
        except Exception as e:
            return f"分析比對結果時出錯: {str(e)}"
    
    def _analyze_with_official_api(self, summary_text):
        """使用官方API進行分析"""
        # 準備API請求
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一個專業的文檔比對分析師，請分析以下文檔比對結果，並提供見解和建議。"
                    },
                    {
                        "role": "user",
                        "content": f"請分析以下文檔比對結果，並提供見解和建議。特別關注未匹配的段落，可能的錯誤類型，以及如何改進。\n\n{summary_text}"
                    }
                ]
            },
            "parameters": {
                "result_format": "message"
            }
        }
        
        # 發送API請求
        response = requests.post(self.api_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            analysis = result["output"]["choices"][0]["message"]["content"]
            return analysis
        else:
            return f"API請求失敗: {response.status_code} - {response.text}"
    
    def _analyze_with_free_api(self, summary_text):
        """使用免費API進行分析"""
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "qwen2.5-7b-instruct",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一個專業的文檔比對分析師，請分析以下文檔比對結果，並提供見解和建議。"
                },
                {
                    "role": "user",
                    "content": f"請分析以下文檔比對結果，並提供見解和建議。特別關注未匹配的段落，可能的錯誤類型，以及如何改進。\n\n{summary_text}"
                }
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(self.free_api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                analysis = result["choices"][0]["message"]["content"]
                return analysis
            else:
                return f"免費API請求失敗: {response.status_code} - {response.text}"
        except Exception as e:
            return f"免費API請求出錯: {str(e)}"
    
    def analyze_table_comparison(self, original_table, edited_table, comparison_result):
        """分析表格比對結果並提供見解"""
        try:
            # 準備表格比對結果摘要
            summary = {
                "original_rows": len(original_table),
                "edited_rows": len(edited_table),
                "matched_cells": comparison_result.get("matched_cells", 0),
                "different_cells": comparison_result.get("different_cells", 0),
                "missing_cells": comparison_result.get("missing_cells", 0),
                "added_cells": comparison_result.get("added_cells", 0)
            }
            
            summary_text = f"""
表格比對結果摘要:
- 原始表格行數: {summary['original_rows']}
- 美編後表格行數: {summary['edited_rows']}
- 匹配單元格數: {summary['matched_cells']}
- 不同單元格數: {summary['different_cells']}
- 缺失單元格數: {summary['missing_cells']}
- 新增單元格數: {summary['added_cells']}

原始表格:
{json.dumps(original_table, ensure_ascii=False, indent=2)}

美編後表格:
{json.dumps(edited_table, ensure_ascii=False, indent=2)}

差異詳情:
{json.dumps(comparison_result.get("differences", []), ensure_ascii=False, indent=2)}
"""
            
            # 使用API進行分析
            if self.use_free_api:
                return self._analyze_with_free_api(summary_text)
            else:
                return self._analyze_with_official_api(summary_text)
        
        except Exception as e:
            return f"分析表格比對結果時出錯: {str(e)}"
    
    def generate_summary_report(self, comparison_results, table_comparison_results=None):
        """生成比對結果摘要報告"""
        try:
            # 準備比對結果統計
            stats = {
                "total_paragraphs": len(comparison_results),
                "matched_paragraphs": sum(1 for r in comparison_results if r["similarity"] >= 0.8),
                "unmatched_paragraphs": sum(1 for r in comparison_results if r["similarity"] < 0.8),
                "average_similarity": sum(r["similarity"] for r in comparison_results) / len(comparison_results) if comparison_results else 0
            }
            
            # 添加表格比對結果統計
            table_stats = {}
            if table_comparison_results:
                table_stats = {
                    "total_tables": len(table_comparison_results),
                    "matched_tables": sum(1 for r in table_comparison_results if r["similarity"] >= 0.8),
                    "unmatched_tables": sum(1 for r in table_comparison_results if r["similarity"] < 0.8),
                    "average_table_similarity": sum(r["similarity"] for r in table_comparison_results) / len(table_comparison_results) if table_comparison_results else 0
                }
            
            # 準備報告內容
            report_content = f"""
# 文檔比對摘要報告

## 段落比對統計
- 總段落數: {stats['total_paragraphs']}
- 匹配段落數: {stats['matched_paragraphs']} ({stats['matched_paragraphs']/stats['total_paragraphs']*100:.2f}%)
- 未匹配段落數: {stats['unmatched_paragraphs']} ({stats['unmatched_paragraphs']/stats['total_paragraphs']*100:.2f}%)
- 平均相似度: {stats['average_similarity']:.2f}
"""

            # 添加表格比對統計
            if table_stats:
                report_content += f"""
## 表格比對統計
- 總表格數: {table_stats['total_tables']}
- 匹配表格數: {table_stats['matched_tables']} ({table_stats['matched_tables']/table_stats['total_tables']*100:.2f}%)
- 未匹配表格數: {table_stats['unmatched_tables']} ({table_stats['unmatched_tables']/table_stats['total_tables']*100:.2f}%)
- 平均表格相似度: {table_stats['average_table_similarity']:.2f}
"""
            
            # 添加未匹配段落列表
            unmatched_paragraphs = [r for r in comparison_results if r["similarity"] < 0.8]
            if unmatched_paragraphs:
                report_content += """
## 未匹配段落列表
"""
                for i, para in enumerate(unmatched_paragraphs[:10]):  # 只顯示前10個
                    report_content += f"""
### 未匹配段落 {i+1}
- 相似度: {para['similarity']:.2f}
- 原始文本: {para['original_text'][:100]}...
- 美編後文本: {para['edited_text'][:100]}...
"""
            
            # 使用API生成報告
            if self.use_free_api:
                return self._generate_report_with_free_api(report_content)
            else:
                return self._generate_report_with_official_api(report_content)
        
        except Exception as e:
            return f"生成摘要報告時出錯: {str(e)}"
    
    def _generate_report_with_official_api(self, report_content):
        """使用官方API生成報告"""
        # 準備API請求
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一個專業的文檔比對報告生成器，請基於以下比對結果生成一份詳細的報告。"
                    },
                    {
                        "role": "user",
                        "content": f"請基於以下比對結果生成一份詳細的報告，包括總體評估、主要問題分析、改進建議等。\n\n{report_content}"
                    }
                ]
            },
            "parameters": {
                "result_format": "message"
            }
        }
        
        # 發送API請求
        response = requests.post(self.api_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            report = result["output"]["choices"][0]["message"]["content"]
            return report
        else:
            return f"API請求失敗: {response.status_code} - {response.text}"
    
    def _generate_report_with_free_api(self, report_content):
        """使用免費API生成報告"""
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "qwen2.5-7b-instruct",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一個專業的文檔比對報告生成器，請基於以下比對結果生成一份詳細的報告。"
                },
                {
                    "role": "user",
                    "content": f"請基於以下比對結果生成一份詳細的報告，包括總體評估、主要問題分析、改進建議等。\n\n{report_content}"
                }
            ],
            "temperature": 0.3,
            "max_tokens": 3000
        }
        
        try:
            response = requests.post(self.free_api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                report = result["choices"][0]["message"]["content"]
                return report
            else:
                return f"免費API請求失敗: {response.status_code} - {response.text}"
        except Exception as e:
            return f"免費API請求出錯: {str(e)}"
