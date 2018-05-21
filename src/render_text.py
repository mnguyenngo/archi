def render_text(doc, wiki_dict):
    """Replaces keywords in user query with html to highlight the keyword"""
    output = doc.text
    possible_comp_dict = find_comp_dict(doc)
    for comp_name, comp_node in wiki_dict.items():
        noun_chunk = search_dict_by_val(possible_comp_dict, comp_name)
        # comp_name_for_url = "_".join(comp_name.split())
        output = output.replace(
            noun_chunk.text, '<a id="noun-chunk" href="/component/{}">'.format(comp_name) + noun_chunk.text + '</a>')
    return output

    # DELETE if code above is OK
    # for chunk in doc.noun_chunks:
    #     chunk_children = list(chunk.root.children)
    #     if len(chunk_children) > 0:
    #     if chunk.root.lemma_ in comp_dict.keys():
    #         output = output.replace(
    #             chunk.text, '<a id="noun-chunk" href=/component/{{ comp }}>' + chunk.text + '</a>')
    # return output


def find_components(doc):
    """Returns the components that are in the knowledge base"""
    components = []
    for chunk in doc.noun_chunks:
        components.append(chunk.root.lemma_)
        chunk_children = list(chunk.root.children)
        if len(chunk_children) > 0:
            for child in chunk_children:
                components.append(" ".join([child.lemma_, chunk.root.lemma_]))

    exclude_list = (['-PRON-',
                     'application',
                     'area',
                     'building',
                     'calculation',
                     'capacity',
                     'consideration',
                     'construction',
                     'design',
                     'inspection',
                     'installation',
                     'load',
                     'material',
                     'method',
                     'procedure',
                     'provision',
                     'result',
                     'requirement',
                     'space',
                     'standard',
                     'structure',
                     'system',
                     'us',
                     'use'])

    components = [comp for comp in components if comp not in exclude_list]

    return components


def find_comp_dict(doc):
    """Returns the components that are in the knowledge base"""
    comp_dict = {}
    for chunk in doc.noun_chunks:
        comp_dict[chunk] = [chunk.root.lemma_]
        chunk_children = list(chunk.root.children)
        if len(chunk_children) > 0:
            for child in chunk_children:
                comp_dict[chunk].append(" ".join([child.lemma_, chunk.root.lemma_]))

    return comp_dict


def search_dict_by_val(d, x):
    for k, v in d.items():
        if x in v:
            return k
