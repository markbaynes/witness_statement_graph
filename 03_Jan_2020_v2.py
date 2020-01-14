import pandas as pd
import spacy
import numpy as np
import re
import csv
import networkx as nx
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# Need this for NetworkX visualisation in Jupyter Notebook
# get_ipython().magic('matplotlib inline')

# USE CONDA PYTHON


from spacy.lang.en import English
nlp = English()  # just the language with no model
nlp = spacy.load("en_core_web_lg")
# nlp.add_pipe(sentencizer, before="parser")

# The Spacy.io bit
def spacy_parse_01(parsed_dict):
    # print(parsed_dict)
    for s_dict in parsed_dict["document_dicts"]:
        for key, value in s_dict.items():
            # print(key, value)
            if key == "string_text":
                doc = nlp(value)
                if len(doc.ents) == 0:
                    # print("No entities found.")
                    break
                elif len(doc.ents) != 0:
                    # found_ents = []
                    for ent in doc.ents:
                        ent_found = [ent.text, ent.label_]
                        s_dict["string_entities"].append(ent_found)
    return parsed_dict 

def create_main_dict():
    """Reads in text file and carries out ent_unique parsing."""

    def replace_foreign_characters(in_string):
        return re.sub(r'[^\x00-\x7f]',r'', in_string)

    def extract_bundle_refs(line):
        """ Extracts bundle references from given string (line) then calls clean_bundle_refs 
        to clean the references. """
        b_refs = re.compile(r"(\(\#\d*\))|(\#\d*\s*)|(\(\#\d*[-]\s*\d*\))")
        found_refs = []
        for m in b_refs.finditer(line):
            found_refs.append(m.group())
        return clean_bundle_refs(found_refs, line)

    def clean_bundle_refs(found_refs, line):
        """ Strips line of bundle references (cleaned_line) then cleans up the found
        references, returns cleaned bundle references and cleaned string. """
        extracted_refs = []
        cleaned_refs = []
        cleaned_line = re.sub(r"(\(\#\d*\))|(\#\d*\s*)|(\(\#\d*[-]\s*\d*\))", '', line)

        for _ in found_refs:
            x = [int(s) for s in re.findall(r'\b\d+\b', _)]
            extracted_refs.append(x)
            for y in extracted_refs:
                for x in y:
                    if x not in cleaned_refs:
                        cleaned_refs.append(x)
                    else:
                        continue
            cleaned_refs.sort()
        return cleaned_refs, cleaned_line

    def get_para_number(text_element):
        """ Processes first line of paragraph text. """                           #
        p_pattern = re.compile(r'^\d*')
        match = re.search(p_pattern, text_element)
        stripped_s = re.sub(p_pattern, '', text_element)
        para_num = match.group(0)
        return para_num

    def header_change(block_text, header_num):
        """ Checks if string 'Paragraph' present, if so processes. """
        header_num += 1
        h_pattern = re.compile(r'^\\Paragraph')
        if re.match(h_pattern, block_text) is None:
            return header_num

    # TESTING PURPOSES ONLY
    filepath = "data/m_edmunds_spell_checked.txt"
    # filepath = "data/m_edmunds_medium.txt"
    # filepath = "data/edmunds_short.txt"

    with open(filepath, encoding ="utf-8") as f:
        statement_dict = {"document_dicts":[]}
        content = f.readlines()
        string_num = 0
        # This needs to be dynamic.
        header_num = 1000
        # Call this on text strings in dict.
        for block in content:
            block_text = block.rstrip("', /\n")
            replace_foreign_characters(block_text)
            # CASE: Check if para header which acts as a header. Will only return result if header.
            result = block_text.find("Paragraph")
            if result == 0:
                header_num = header_change(block_text, header_num)
            elif result == -1:
                # CASE: First string in block_text with para number. ")
                """ Processes first line of paragraph text. """
                pattern = re.compile(r'^\d*')
                match = re.search(pattern, block_text)
                this_text = re.sub(pattern, '', block_text)
                para_num = match.group(0)
                doc = nlp(this_text)
                para_text = [sent.string.strip(".") for sent in doc.sents]
                for sentence in para_text:
                    sentence = sentence.strip()
                    cleaned_refs, cleaned_line = extract_bundle_refs(sentence)
                    string_num += 1
                    string_dict = {"header_num":header_num, 'para_num': para_num, 'string_num': string_num, 'id': 0, 'string_text': cleaned_line, "string_bundle_refs": cleaned_refs, 'string_entities': []}
                    statement_dict["document_dicts"].append(string_dict)
        return statement_dict

def identify_unique_entities(parsed_dict):
    """ Utility to generate list of unique entities found
    and save as CSV. """
    unique_entities = []
    entity_id = 0
    for x in parsed_dict["document_dicts"]:
        for k, v in x.items():
            if k == "string_entities":
                for this_entity in v:
                    if this_entity in unique_entities:
                        continue
                    elif this_entity not in unique_entities:
                        unique_entities.append(this_entity)

    for target_ent in unique_entities:
        entity_id += 1
        target_ent.insert(0,entity_id)
    return unique_entities

def update_entity_ids(parsed_dict, unique_entities):
    dict_num = 0
    # TODO NOT RUNNING THROUGH LIST AGAIN?
    for x in parsed_dict["document_dicts"]:
        for key, value in x.items():
            # print(key, value)
            if key == "string_entities":
                if not value:
                    break
                else:
                    for ent in unique_entities:
                        for i, y in enumerate(value):
                            if y == ent:
                                break
                            elif len(y) <= 2 and y[0:2] == ent[1:3]:
                                # print(f"{y[0:2]} PARTIAL {ent}")
                                value.remove(y)
                                # print(f"POP {y}")
                                value.append(ent)
                                # print(f"PUSH {ent}")
                                # print(f"UPDATED {value}\n")

    return parsed_dict


def calculate_entity_nodes(parsed_dict):
    """ Generate list of entity nodes. """
    entity_nodes = []
    header_count = 0

    for x in parsed_dict["document_dicts"]:
        if header_count == x["header_num"]:
            pass
        for y in x["string_entities"]:
            # Length check to catch entities that have not been properly processed.
            if len(y) <= 2:
                break
            else:
                id = y[0] 
                label = y[1]
                node_type = y[2]
                ent_node = [id, label, node_type]
                entity_nodes.append(ent_node)
    return entity_nodes

def calculate_string_nodes(parsed_dict, ent_count):
    """ Generate list of string nodes. """
    string_nodes = []
    id = ent_count
    for x in parsed_dict["document_dicts"]:
        id += 1
        x.update({"id": id})
        label = "String" + "_" + str(x["string_num"])
        node_type = "String"
        string_node = [id, label, node_type]
        string_nodes.append(string_node)
    return string_nodes


# TODO everything unique. 
def calculate_all_edges(parsed_dict):
    """Creates index of unique entities locations. """
    all_edges = []
    for x in parsed_dict["document_dicts"]:
        for y in x["string_entities"]:
            source = x["id"] # Retrieve unique string ID.
            # Length check to catch entities that have not been properly processed.
            if len(y) <= 2:
                break
            else:
                target = y[0]
                this_edge = [source, target]
                all_edges.append(this_edge)
                # Undirected graph so...
                this_edge = [target, source]
                all_edges.append(this_edge)
                # Reversed edge too?
                # print(this_edge)
    return all_edges

# NOT DOING ALL EDGES

def check_graph_validity():
    print("Exporting graph structure...")
    G = nx.Graph()
    nodes = pd.read_csv ("data/output/entity_nodes.csv")
    df_string_nodes = pd.read_csv ("data/output/df_string_nodes.csv")
    all_edges = pd.read_csv ("data/output/all_edges.csv")
    # print(all_edges)
    # Dataframe to list
    nodes_list = nodes.values.tolist()
    string_nodes_list = df_string_nodes.values.tolist()
    all_edges_list = all_edges.values.tolist()
    # Import id, name, and group into node of Networkx
    for i in nodes_list:
        G.add_node (i[0], name=i[1], group=i[2])
    # Import id, name, and group into node of Networkx
    for i in string_nodes_list:
        G.add_node (i[0], name=i[1], group=i[2])
    # Import source, target into all_edges of Networkx
    for i in all_edges_list:
        G.add_edge (i[0], i[1])
    # Visualize the network:
    nx.draw_networkx (G)
    print ("Validating graph - hang on a moment...")
    # plt.show (block=True)
    nx.write_graphml (G, "data/output/simple_v2.graphml")
    print ("Graph validated.")


def main():
    '''Main function.
    Reads in txt file, parses it, then processes using spacy.io library.'''
    print("Running... wait a bit.")
    statement_dict = create_main_dict()
    print("Doing NLP with spacy...")
    nlp_dict = spacy_parse_01(statement_dict)
    print(nlp_dict)
    # Create list of unique ents
    # unique_entities = identify_unique_entities(nlp_dict)

    # df_ents = pd.DataFrame(unique_entities)
    # df_ents.columns = ["id", "label","node_type"]
    # df_ents.to_csv("data/output/unique_entities.csv", index=False)
    # print("Unique entities CSV written...")
    # parsed_dict = update_entity_ids(nlp_dict, unique_entities)
    # columns = ["header_num", "para_num", "string_num", "id",'string_text', "string_bundle_refs",  'string_entities']
    # df_parsed_dict = pd.DataFrame.from_dict(parsed_dict["document_dicts"], orient='columns')
    # df_parsed_dict.to_csv("data/output/df_parsed_dict.csv", index=False)
    # print("Parsed dict DF written to CSV...") 
    

    # Generate entity nodes.
    # entity_nodes = calculate_entity_nodes(parsed_dict)
    # # print(f"Entity nodes {entity_nodes}")
    # df_entity_nodes = pd.DataFrame(entity_nodes)
    # df_entity_nodes.columns = ["id", "label", "node_type"]
    # df_entity_nodes.to_csv("data/output/entity_nodes.csv", index=False)
    # print("Entity nodes CSV written...")
    # # Pass the entity ID number through
    # # ent_count = df_entity_nodes['id'].max()
    # # ent_count += 1
    # # Generate string nodes.
    # string_nodes = calculate_string_nodes(parsed_dict, df_entity_nodes['id'].max())
    # columns = ["header_num", "para_num", "string_num", "id",'string_text', "string_bundle_refs",  'string_entities']
    # df_parsed_dict = pd.DataFrame.from_dict(parsed_dict["document_dicts"], orient='columns')
    # df_parsed_dict.to_csv("data/output/df_parsed_dict.csv", index=False)
    # print("Parsed dict updated DF written to CSV...") 
    # # print(f"String nodes {string_nodes}")
    # df_string_nodes = pd.DataFrame(string_nodes)
    # df_string_nodes.columns = ["id", "label", "node_type"]
    # df_string_nodes.to_csv("data/output/df_string_nodes.csv", index=False)
    # print("String nodes CSV written...")


    # Generate all_edges.

    # all_edges = calculate_all_edges(parsed_dict)
    # # print(f"Edges {all_edges}")
    # df_all_edges = pd.DataFrame(all_edges)
    # df_all_edges.columns = ["source", "target"]
    # df_all_edges.to_csv("data/output/all_edges.csv", index=False)
    # print("Edges written...")
    
    # check_graph_validity()
    print("Run complete.")


if __name__ == '__main__':
    main()
     