def render_text(doc):
    output = doc.text
    for chunk in doc.noun_chunks:
        output = output.replace(
            chunk.text, '<span id="noun-chunk">' + chunk.text + '</span>')
    return output
