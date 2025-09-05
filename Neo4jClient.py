from neo4j import GraphDatabase

class Neo4jClient:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def execute_query(self, query):
        """
        执行 Cypher 查询并返回结果。
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [record for record in result]

import pandas as pd
import networkx as nx

class KnowledgeGraphBuilder:
    def __init__(self, csv_file):
        self.csv_file = csv_file

    def build_tree_and_classify(self):
        """
        构建知识图谱树并分类节点。
        """
        mapp = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六"}
        df = pd.read_csv(self.csv_file, header=None, names=['Entity1', 'Relation', 'Entity2'], skiprows=1)

        graph = nx.DiGraph()
        for _, row in df.iterrows():
            graph.add_edge(row['Entity1'], row['Entity2'])

        depths = {}
        root_nodes = [n for n in graph.nodes if graph.in_degree(n) == 0]
        for root in root_nodes:
            for node, depth in nx.single_source_shortest_path_length(graph, root).items():
                depths[node] = depth

        nodes = []
        for node, depth in depths.items():
            label = f"{mapp[depth + 1]}级知识点"
            nodes.append({"name": node, "label": label})

        relationships = df.to_dict('records')

        return nodes, relationships

from openai import OpenAI

class OpenAIClient:
    def __init__(self, base_url, api_key):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate_query(self, prompt):
        """
        使用 OpenAI API 生成查询语句。
        """
        stream = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            stream=True
        )
        query = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                query += chunk.choices[0].delta.content
        return query.strip()

class PromptGenerator:
    @staticmethod
    def generate_recursive_prompt(user_question, entity_types, relation_types):
        """
        生成递归查询的 Prompt。
        """
        kg_info = (
            "### 知识图谱背景\n"
            f"实体种类：{', '.join(entity_types)}\n"
            f"关系种类：{', '.join(relation_types)}\n"
            f"知识图谱的结构是一个有向图，关系方向只能从上到下。\n"
        )

        examples = [
            {
                "question": "人工智能的可解释性有哪些方法？",
                "entity": "人工智能的可解释性",
                "query": 'MATCH path = (a {name: "人工智能的可解释性"})-[:有|包括|定义|包含*]->(b) '
                         'RETURN DISTINCT nodes(path) AS all_nodes, relationships(path) AS all_relationships'
            },
            {
                "question": "人工智能伦理的定义是什么？",
                "entity": "人工智能伦理",
                "query": 'MATCH path = (a {name: "人工智能伦理"})-[:有|包括|定义|包含*]->(b) '
                         'RETURN DISTINCT nodes(path) AS all_nodes, relationships(path) AS all_relationships'
            },
            {
                "question": "隐私保护技术有哪些相关节点和关系？",
                "entity": "隐私保护技术",
                "query": 'MATCH path = (a {name: "隐私保护技术"})-[:有|包括|定义|包含*]->(b) '
                         'RETURN DISTINCT nodes(path) AS all_nodes, relationships(path) AS all_relationships'
            }
        ]

        example_text = "\n\n".join([
            f"### 示例 {i + 1}\n输入问题：{ex['question']}\n提取结果：\n- 实体：{ex['entity']}\n生成的查询语句：\n{ex['query']}"
            for i, ex in enumerate(examples)
        ])

        return f"""
        你是一个自然语言解析器，可以从输入的问题中提取目标实体，并生成针对知识图谱的查询语句。以下是当前知识图谱的背景信息：

        {kg_info}

        以下是几个示例：

        {example_text}

        ### 任务
        现在，我将输入一个问题，请按照上述格式提取实体，并生成查询语句。在输出中请只包括查询语句。

        输入问题：{user_question}
        """

    @staticmethod
    def generate_description_prompt(formatted_data):
        """
        生成描述性文本的 Prompt。
        """
        return f"""
        你是一个知识图谱解析器，能够根据实体和关系生成描述性文本。以下是知识图谱的背景信息：

        ### 知识图谱背景
        知识图谱由实体和关系组成：
        - 实体：是知识的节点，例如 "人工智能的可解释性"、"隐私保护技术"。
        - 关系：是连接实体的边，例如 "定义"、"包括"、"有"。

        每个关系连接两个实体，表示它们之间的知识关联。

        ### 示例
        #### 输入
        实体和关系：
        - 实体 1：人工智能的可解释性
        - 实体 2：神经网络中层特征的可解释性
        - 关系：有

        #### 输出
        "人工智能的可解释性包含了一个重要方面：神经网络中层特征的可解释性，这是对神经网络中层特征语义明确性的深入探讨。"

        #### 输入
        实体和关系：
        - 实体 1：隐私保护技术
        - 实体 2：差分隐私
        - 关系：包括

        #### 输出
        "隐私保护技术的一个重要组成部分是差分隐私，这种技术通过添加噪声来保护用户数据。"

        ### 任务
        现在，我将输入一组实体和关系，请根据这些信息生成一段描述性文本。

        #### 输入
        {formatted_data}

        #### 输出
        请生成描述性文本：
        """

if __name__ == "__main__":
    # 初始化 Neo4j 客户端
    neo4j_client = Neo4jClient(uri="bolt://localhost:7687", username="neo4j", password="Mitsuha20040828")

    # 初始化知识图谱构建模块
    kg_builder = KnowledgeGraphBuilder(csv_file="detailed_knowledge_graph.csv")
    nodes, relationships = kg_builder.build_tree_and_classify()

    # 初始化 OpenAI 客户端
    openai_client = OpenAIClient(base_url="https://hk.soruxgpt.com/api/api/v1", api_key="sk-ipOShdrs3sZgMQ73lXT8uI4bvF0Hh6sE1SCkN5iJ186tV9B3")

    # 生成 Prompt 并查询
    entity_types = [node["label"] for node in nodes]
    relation_types = list(set(rel["Relation"] for rel in relationships))
    user_question = "人工智能的可解释性有哪些方法？"

    recursive_prompt = PromptGenerator.generate_recursive_prompt(user_question, entity_types, relation_types)
    query = openai_client.generate_query(recursive_prompt)
    print(f"生成的查询语句：\n{query}")

    # 执行 Neo4j 查询
    results = neo4j_client.execute_query(query)

    # 解析查询结果
    formatted_data = [

    ]
    for result in results:
        rel = (result.get("all_relationships"))
        entity1 = rel[0].nodes[0]._properties['name']
        entity2 = rel[0].nodes[1]._properties['name']
        relation = rel[0].type
        formatted_data.append({"entity_1":entity1, "entity_2":entity2, "relation":relation})
    # 生成描述性文本的 Prompt
    description_prompt = PromptGenerator.generate_description_prompt(formatted_data)
    description = openai_client.generate_query(description_prompt)
    print(f"生成的描述性文本：\n{description}")

    # 关闭 Neo4j 客户端
    neo4j_client.close()
