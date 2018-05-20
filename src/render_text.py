def render_text(doc, comp_dict):
    output = doc.text
    for chunk in doc.noun_chunks:
        if chunk.root.lemma_ in comp_dict.keys():
            output = output.replace(
                chunk.text, '<span id="noun-chunk">' + chunk.text + '</span>')
    return output

def find_components(doc):
    components = []
    for chunk in doc.noun_chunks:
        components.append(chunk.root.lemma_)
        chunk_children = list(chunk.root.children)
        if len(chunk_children) > 0:
            for child in chunk_children:
                components.append(" ".join([child.lemma_, chunk.root.lemma_]))
    return components
