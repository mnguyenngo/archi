import archi_nlp


def main():
    print('Creating Archi')
    archi = archi_nlp.Archi()
    print('Getting nlp data')
    archi.get_nlp_data('data/nlp_df/nlp_0514.pkl')
    print('Starting mongo database')
    archi.start_mongo()
    print('Adding data to database')
    archi.build_db()


if __name__ == '__main__':
    main()
