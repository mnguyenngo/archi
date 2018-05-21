from pymongo import MongoClient
import requests


def get_wiki_content(name):
    wp_base_url = 'http://en.wikipedia.org/w/api.php'
    params = {'prop': 'pageprops',
              'action': 'query',
              'format': 'json',
              'titles': name}
    resp = requests.get(wp_base_url, params=params)
    data = resp.json()
    wp_url, wd_url, wd_id = None, None, None
    if 'query' in data.keys():
        keys = data['query']['pages'].keys()
        check = '-1' not in keys  # check if wikipedia for 'name' exists
        if check:
            name = name.replace(" ", "_")
            wp_url = "https://en.wikipedia.org/wiki/{}".format(name)

            wiki_pages_values = list(data['query']['pages'].values())[0]
            if 'pageprops' in wiki_pages_values.keys():
                try:
                    wd_id = wiki_pages_values['pageprops']['wikibase_item']
                    wd_url = "http://www.wikidata.org/entity/{}".format(wd_id)
                    print(wd_id)
                except:
                    pass

    return wp_url, wd_url, wd_id


def add_wp_url(coll):
    components = list(coll.find({'@type': 'component'}))

    for comp in components:
        wp_url, wd_url, wd_id = get_wiki_content(comp['name'])
        if wp_url is not None:
            coll.update(
                {'name': comp['name']},
                {'$set': {
                    'wikipedia': wp_url,
                    'wikidata_url': wd_url,
                    'wikidata_id': wd_id}},
                upsert=True)
            print("Added wikipedia url for {}".format(comp['name']))


if __name__ == '__main__':
    client = MongoClient()
    db = client['archi']
    coll = db['archi_180521']

    add_wp_url(coll)
