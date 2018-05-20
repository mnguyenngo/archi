import archi_nlp


def main():
    """For EC2 instance
    Creates archi object, starts a local mongo db and adds data to the db.
    """
    print('Creating Archi...')
    archi = archi_nlp.Archi()
    print('Getting nlp data...')
    archi.get_nlp_data('data/nlp_df/nlp_0514.pkl')
    print('Starting mongo database...')
    archi.start_mongo()
    print('Adding data to database...')
    archi.build_db()


if __name__ == '__main__':
    main()
