import os
from rdflib import Graph
from fuzzywuzzy import fuzz
from common import logger


g = Graph()
g.parse(os.path.join("..", "data", "knowledge_graph", "companies.ttl"), format="ttl")


def is_company_match(source, company_name):
    query = """
    PREFIX ex: <http://example.com/>

    SELECT ?altLabel
    WHERE {{
        ?company ex:sourceDoc ?sourceDoc ;
                ex:altLabel ?altLabel .
        FILTER(?sourceDoc = "{}")
    }}
    """.format(source)

    results = g.query(query)

    logger.info(source)
    for row in results:
        if fuzz.ratio(company_name.lower(), row.altLabel.lower()) > 75:
            return True
    return False
