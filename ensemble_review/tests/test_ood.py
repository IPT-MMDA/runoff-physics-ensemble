import torch

from complexity_ensemble.ood import MahalanobisOODDetector


def test_mahalanobis_detector_flags_a_distant_window():
    training = torch.tensor([
        [[-0.1, 0.0], [0.0, 0.1]],
        [[0.0, -0.1], [0.1, 0.0]],
        [[0.05, 0.0], [0.0, -0.05]],
        [[-0.05, 0.05], [0.05, -0.05]],
    ])
    detector = MahalanobisOODDetector(quantile=0.9, shrinkage=0.2).fit(training)
    central_index = int(detector.score(training).argmin())
    central = training[central_index:central_index + 1]
    test = torch.cat((central, torch.full_like(training[:1], 10.0)))

    flags = detector.is_ood(test)

    assert flags.tolist() == [False, True]
    assert detector.score(test)[1] > detector.score(test)[0]
