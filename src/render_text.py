def render_text(doc, comp_dict):
    """Replaces keywords in user query with html to highlight the keyword"""
    output = doc.text
    for chunk in doc.noun_chunks:
        if chunk.root.lemma_ in comp_dict.keys():
            output = output.replace(
                chunk.text, '<span id="noun-chunk">' + chunk.text + '</span>')
    return output


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
                     'standard',
                     'structure',
                     'system',
                     'us',
                     'use'])

    components = [comp for comp in components if comp not in exclude_list]

    return components
