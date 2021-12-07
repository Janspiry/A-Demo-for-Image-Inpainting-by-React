import React from 'react';

import {TableRow, TableRowColumn} from 'material-ui';
import {scaleSequential} from 'd3-scale';
import {interpolateInferno} from 'd3-scale-chromatic'


const SCALE = scaleSequential(interpolateInferno).domain([0,1]);

export function drawImage(ctx, src, callback) {
    const img = new Image(512, 512);
    img.src = src;

    img.onload = function () {
        ctx.clearRect(0, 0, 512, 512);
        ctx.drawImage(img, 0, 0);
        callback(img);
    }
}



export async function inpaint(iCtx, dCtx) {
    const mask = dCtx.getImageData(0, 0, 512, 512);
    const img = iCtx.getImageData(0, 0, 512, 512);

    const mask_u8 = new Uint8Array(512 * 512);
    for(let n = 0; n < mask.data.length; n+=4){
        if (mask.data[n] > 0) {
            mask_u8[n/4] = 1;
        } else {
            mask_u8[n/4] = 0;
        }
    };

    return fetch('http://127.0.0.1:5000/inpaint', {
        method: 'POST',
        mode: 'cors',
        headers: {
            'Accept': 'application/json',
        },
        body: JSON.stringify({
            "image": Array.from(img.data),
            "mask": Array.from(mask_u8)
        })
    }).then(res => res.json())
    .then(res => {
        const processedImg = new ImageData(Uint8ClampedArray.from(res), 512, 512);
        return [processedImg, null];
    })
    .catch(err => {
        console.log("ERROR:", err, "Inpaint Error");
        return null;
   });
}

export async function changeModel(image_type, mask_mode) {
    return fetch('http://127.0.0.1:5000/changeModel', {
        method: 'POST',
        mode: 'cors',
        headers: {
            'Accept': 'application/json',
        },
        body: JSON.stringify({
            'image_type': image_type,
            "mask_mode": mask_mode
        })
    }).then(res => res.json())
    .then(res => {
        console.log("Info:", "Changed the model");
        return true
    })
    .catch(err => {
        console.log("ERROR:", err, "Changed Error");
        return false
   });
}


export async function randomImage(image_type, mask_mode) {
    return fetch('http://127.0.0.1:5000/randomImage', {
        method: 'POST',
        mode: 'cors',
        headers: {
            'Accept': 'application/json',
        },
        body: JSON.stringify({
            'image_type': image_type,
            "mask_mode": mask_mode
        })
    }).then(res => res.json())
    .then(res => {
        const randomImage = new ImageData(Uint8ClampedArray.from(res['image']), 512, 512);
        const randomResult = new ImageData(Uint8ClampedArray.from(res['result']), 512, 512);
        const randomMask = new ImageData(Uint8ClampedArray.from(res['mask']), 512, 512);
        return [randomImage, randomMask, randomResult, null];
    })
    .catch(err => {
        console.log("ERROR:", err, "randomImage Error");
        return null;
   });
}
export default drawImage;