from neo4j import GraphDatabase
import re
import requests

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query(self, query, parameters=None, db=None):
        with self.driver.session(database=db) as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]  # 确保返回的是字典

def clean_entity_name(entity):
    """移除标点符号和空格"""
    return re.sub(r'[^\w\s]', '', entity).strip()

def parse_question(question):
    """将自然语言问题转换为Cypher查询"""
    if "什么是" in question or "介绍" in question:
        entity = question.replace("什么是", "").replace("介绍", "").strip()
        entity = clean_entity_name(entity)
        query = f"MATCH (n:Topic {{name: '{entity}'}}) RETURN n.description AS description"
    elif "图片" in question or "图像" in question:
        entity = re.split(r'的图片', question)[0].strip()
        entity = clean_entity_name(entity)
        query = f"MATCH (n:Topic {{name: '{entity}'}})-[:HAS_IMAGE]->(img:Image) RETURN img.image AS image_url"
    elif "关系" in question:
        entity = re.split(r'的关系', question)[0].strip()
        entity = clean_entity_name(entity)
        query = f"MATCH (n:Topic {{name: '{entity}'}})-[r:RELATED_TO]->(m) RETURN m.name AS related_topics"
    else:
        query = None
    return query

def get_answer(conn, question):
    """通过Cypher查询获取答案"""
    query = parse_question(question)
    if query:
        result = conn.query(query)
        if result:
            if 'description' in result[0]:
                return result[0]['description']  # 返回description内容
            elif 'image_url' in result[0]:
                return result[0]['image_url']  # 返回图像链接
            elif 'related_topics' in result[0]:
                return [r['related_topics'] for r in result]  # 返回所有相关的主题
        return "对不起，未找到相关信息。"
    return "我不确定如何回答这个问题。"

def extract_keywords(neo4j_answers):
    """从Neo4j回答中提取关键词"""
    keywords = []
    for answer in neo4j_answers:
        # 跳过以http/https开头的答案
        if answer is None or not isinstance(answer, str) or answer.startswith(('http://', 'https://')):
            continue
        words = re.split(r'[\s,，.。]+', answer)
        keywords.extend([w for w in words if w])
    return keywords

def llmchat(question):
    """大模型请求"""
    host = "http://localhost"
    port = "11434"
    model = "llama3"
    url = f"{host}:{port}/api/chat"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "options": {"temperature": 0},
        "stream": False,
        "messages": [{"role": "system", "content": question}]
    }
    response = requests.post(url, json=data, headers=headers, timeout=600)
    if response.status_code == 200:
        answer = response.json().get("message").get('content')
    else:
        answer = "请求失败，未获得答案。"
    return answer

def save_to_file(text, filename="answers.txt"):
    """将文本保存到文件"""
    with open(filename, "a", encoding="utf-8") as file:
        file.write(text + "\n\n")  # 每个回答后加换行符分隔

if __name__ == "__main__":
    conn = Neo4jConnection(uri="bolt://localhost:7687", user="neo4j", password="12345678")

    question1 = "什么是人工智能伦理与安全"
    question2 = "后门攻击的图片"
    question3 = "人工智能伦理与安全的关系"

    # 获取Neo4j查询结果
    answer1 = get_answer(conn, question1)
    answer2 = get_answer(conn, question2)
    answer3 = get_answer(conn, question3)

    print(f"Q: {question1}\nA: {answer1}\n")
    print(f"Q: {question2}\nA: {answer2}\n")
    print(f"Q: {question3}\nA: {answer3}\n")

    # 将Neo4j查询的结果保存到文件
    save_to_file(f"Q: {question1}\nA: {answer1}")
    save_to_file(f"Q: {question2}\nA: {answer2}")
    save_to_file(f"Q: {question3}\nA: {answer3}")

    # 提取关键词并向大模型提问
    neo4j_answers = [answer1, answer2, answer3]
    keywords = extract_keywords(neo4j_answers)

    # 针对 http/https 直接提问大模型
    for answer in neo4j_answers:
        if isinstance(answer, str) and answer.startswith(('http://', 'https://')):
            question = f"关于这个链接的详细信息是什么？{answer}"
            print(f"向大模型提问: {question}")
            llm_answer = llmchat(question)
            print(f"大模型回答: {llm_answer}\n")
            save_to_file(f"Q: {question}\nA: {llm_answer}")

    # 使用从文本中提取的关键词向大模型提问
    for keyword in keywords:
        question = f"关于{keyword}的详细信息是什么？"
        print(f"向大模型提问: {question}")
        llm_answer = llmchat(question)
        print(f"大模型回答: {llm_answer}\n")
        save_to_file(f"Q: {question}\nA: {llm_answer}")

    # 针对 '人工智能伦理与安全的关系' 提取的相关主题向大模型提问
    if isinstance(answer3, list):  # 如果第三个问题的答案是列表
        for topic in answer3:
            question = f"关于{topic}的详细信息是什么？"
            print(f"向大模型提问: {question}")
            llm_answer = llmchat(question)
            print(f"大模型回答: {llm_answer}\n")
            save_to_file(f"Q: {question}\nA: {llm_answer}")

    conn.close()
