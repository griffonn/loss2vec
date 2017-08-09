from nltk.corpus import wordnet as wn


def get_syn(words):
    return {
        w: list(
            set(
                sum(
                    list(
                        map(lambda ss: list(map(lambda l: l.name(), ss.lemmas())),
                            wn.synsets(w))),
                    []
                )
            )
        )
        for w in words
        }


def get_ant(words):
    return {
        w: list(
            set(
                sum(
                    list(
                        map(lambda ss: sum(list(map(lambda l: list(map(lambda a: a.name(), l.antonyms())), ss.lemmas())),
                                           []),
                            wn.synsets(w))),
                    []
                )
            )
        )
        for w in words
        }