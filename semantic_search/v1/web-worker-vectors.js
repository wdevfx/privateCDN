import { pipeline, cos_sim } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.6.0';

const initModel = async () => await pipeline(
    'feature-extraction',
    'Xenova/all-MiniLM-L6-v2',
    { dtype: 'q8' },
);

const getVectorString = (item) =>
    Object.entries(item)
        .map(([key, value]) => `${key}:${value}`)
        .join(' ');

const prepareVectors = async (dataArray) => {
    const getEmbeddedVector = await initModel();
    const vectorMap = {};

    for (let idx = 0; idx < dataArray.length; idx++) {
        const { id, ...restValues } = dataArray[idx];
        const str = getVectorString(restValues);
        const rawVector = await getEmbeddedVector(str, { pooling: 'mean', normalize: true });

        vectorMap[id] = { vector: Array.from(rawVector.data), data: str };
    }

    return vectorMap;
}

const getQueryVector = async (query) => {
    const getEmbeddedVector = await initModel();
    return Array.from((await getEmbeddedVector(query, { pooling: 'mean', normalize: true })).data);
}

const getScoredData = (vectorsArray, queryVector) => {
    return vectorsArray.reduce((result, [id, { vector }]) => {
        result[id] = cos_sim(queryVector, vector);
        return result;
    }, {});
}

const getSortedDataByScores = (dataArray, key, scoresMap) =>
    dataArray.sort((a, b) => scoresMap[b[key]] - scoresMap[a[key]]);


const handleMessage = async (event) => {
    const { data: { command, payload } } = event;

    switch (command) {
        case 'get-data-vectors':
            return {
                command: 'complete',
                payload: await prepareVectors(payload.dataArray),
            };
        case 'get-query-vector':
            return {
                command: 'complete',
                payload: await getQueryVector(payload.constData),
            };
        case 'get-scored-data':
            return {
                command: 'complete',
                payload: getScoredData(payload.dataArray, payload.constData)
            }
        case 'get-sorted-data-by-scores':
            return {
                command: 'complete',
                payload: getSortedDataByScores(
                    payload.constData.dataArray,
                    payload.constData.key,
                    payload.constData.scoresMap,
                ),
            }
        default:
            throw Error(`unknown message: ${command}`);
    }
}

onmessage = async (event) => {
    const result = await handleMessage(event);
    postMessage(result);
}


