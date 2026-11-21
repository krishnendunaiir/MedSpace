from src.split import pid, split


def test_pid():
    assert pid("x/person1_bacteria_1.jpeg") == "person1"
    assert pid("x/person1_virus_6.jpeg") == "person1"
    assert pid("x/IM-0115-0001.jpeg") == "0115"
    assert pid("x/NORMAL2-IM-1427-0001.jpeg") == "1427"


def test_split():
    p = [f"person{i}_bacteria_{j}.jpeg" for i in range(40) for j in range(3)]
    tr, va, te = split(p)
    assert len(tr) + len(va) + len(te) == len(p)

    def g(ix):
        return {pid(p[i]) for i in ix}

    assert not (g(tr) & g(va)) and not (g(tr) & g(te)) and not (g(va) & g(te))
    assert len(va) > 0 and len(te) > 0
