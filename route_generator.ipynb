# Data in thedf, destdf, srcdf
# Map API uses (lat,lon) and PyDeck uses (lon, lat)

import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import requests
np.random.seed(seed=1)

thresh = 0.4
vthresh = 1
lat0 = 48.8566
lon0 = 2.3522
cent = np.array([lon0, lat0])


rho = np.sqrt(np.abs(np.random.normal(0, 0.5, 1000)))
phi = np.random.uniform(0, 2 * np.pi, 1000)

x = rho * np.cos(phi) + lat0
y = rho * np.sin(phi) * 1.5 + lon0
dest = np.column_stack([x, y])


np.random.seed(90)
rho = np.sqrt(np.abs(np.random.normal(0, 0.4, 10)))
phi = np.random.uniform(0, 2 * np.pi, 10)

x = rho * np.cos(phi) + lat0
y = rho * np.sin(phi) * 1.5 + lon0
src = np.column_stack([x, y])



def stack(x, i=15):
    if i == 0:
        return np.row_stack([x, x])
    else:
        return np.row_stack([stack(x, i - 1), stack(x, i - 1)])



def trans(labels, cluster):
    rlabels = np.ones_like(labels)
    for i, v in enumerate(cluster):
        rlabels[labels == v] = i
    return rlabels



ndest = np.row_stack([dest, stack(src)])

kmeans = KMeans(n_clusters=src.shape[0], random_state=0).fit(ndest)
labels = kmeans.labels_[:1000]

labels = trans(labels, list(kmeans.predict(src)))

dsrc = np.column_stack([dest, labels])
dsource = np.array([
    np.column_stack([dsrc[dsrc[:, 2] == i, 0], dsrc[dsrc[:, 2] == i, 1]])
    for i in range(src.shape[0])
])



def near(x, i):
    return np.linalg.norm(dsource[i] - src[i], axis=1) < thresh


nearpoints = [
    np.column_stack(
        [dsource[i][near(dsource, i), 0], dsource[i][near(dsource, i), 1]])
    for i in range(src.shape[0])
]
farpoints = [
    np.column_stack(
        [dsource[i][~near(dsource, i), 0], dsource[i][~near(dsource, i), 1]])
    for i in range(src.shape[0])
]



def getlist(points, n, mode="proto"):
    s = str(src[n][0]) + ',' + str(src[n][1]) + ':'
    if mode == "proto":
        pt = points[n]
    else:
        pt = points
    for i, v in enumerate(pt):
        if i < 145:
            s = s + str(v[0]) + ',' + str(v[1]) + ':'
        else:
            break
    s = s + str(src[n][0]) + ',' + str(src[n][1])
    return s


def getroute(query, routeRepr="summaryOnly"):

    subscriptionKey = "3S9d9XPp-wT2oH-8yDBeVAknA5s2ykqPFCmTWANPgN8"

    #Get boundaries for the electric vehicle's reachable range.
    resp = requests.get(
        "https://atlas.microsoft.com/route/directions/json?subscription-key={}&api-version=1.0&query={}&routeType=shortest&computeBestOrder=true&travelMode=car&routeRepresentation={}"
        .format(subscriptionKey, query, routeRepr)).json()
    return resp



def cluster(points):
    labels = []
    nlabels = []
    clusterpoints = []

    for i in range(src.shape[0]):
        query = getlist(points, i)
        resp = getroute(query)
        length = resp['routes'][0]['summary']['lengthInMeters']

        if type == "far":
            k = (vthresh * length) // 400000
        else:
            k = (vthresh * length) // 200000
        k = int(k)

        if k > 0:
            npoints = np.row_stack([points[i], stack(src[i])])
            size = points[i].shape[0]
            kmeans = KMeans(n_clusters=k, random_state=0).fit(npoints)
            srclabel = kmeans.labels_[size]
            lab = kmeans.labels_[:size]

            nlabels.append(np.unique(lab).size)
            lab = np.where(lab == nlabels[-1], srclabel, lab)
            labels.append(lab)
        else:
            nlabels.append(k)
            print(f"none for {i}")

        path = [
            np.column_stack([points[i][lab == j, 0], points[i][lab == j, 1]])
            for j in range(nlabels[-1])
        ]
        clusterpoints.append(path)

    return clusterpoints


nearclusters = cluster(nearpoints)

farclusters = cluster(farpoints)



def getpath(points):
    paths = []
    for i, srcpt in enumerate(points):
        print(src.shape[0] - i, end="->")
        subpaths = []

        for vehpt in srcpt:
            query = getlist(vehpt, i, mode="veh")
            resp = getroute(query, "polyline")
            data = {}
            data['length'] = resp['routes'][0]['summary']['lengthInMeters']
            data["path"] = []
            for wp in resp['routes'][0]["legs"]:
                data["path"] += wp['points']
            data['optorder'] = resp["optimizedWaypoints"]
            subpaths.append(data)

        paths.append(subpaths)

    return paths


nearpaths = getpath(nearclusters)
farpaths = getpath(farclusters)


def vehdf(paths, src, veh):
    df = pd.DataFrame(paths[src][veh]['path'])
    df = pd.DataFrame({
        "src":
        src,
        "veh":
        veh,
        "path": [df.reset_index()[["longitude", "latitude"]].values.tolist()]
    })
    return df


def srcdf(paths, src):
    df = pd.DataFrame()
    for veh in range(len(paths[src])):
        df = pd.concat([df, vehdf(paths, src, veh)], axis=0)
    df.reset_index(drop=True, inplace=True)
    return df


def maindf(paths):
    df = pd.DataFrame()
    for src in range(len(paths)):
        df = pd.concat([df, srcdf(paths, src)], axis=0)
    df.reset_index(drop=True, inplace=True)
    return df


fulldfnear = maindf(nearpaths)
fulldffar = maindf(farpaths)

fulldf = pd.concat([fulldfnear.assign(type=0), fulldffar.assign(type=1)])
fulldf.reset_index(drop=True, inplace=True)

fulldf.to_json("./fulldf.json")
destdf = pd.DataFrame({
    "labels": labels.tolist(),
    "coordinates": dest.T[[1, 0]].T.tolist()
})
srcdf = pd.DataFrame({
    "labels": range(src.shape[0]),
    "coordinates": src.T[[1, 0]].T.tolist()
})


destdf.to_json("./destdf.json")
srcdf.to_json("./srcdf.json")
