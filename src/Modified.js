import React, { Component } from 'react';

import {Paper} from 'material-ui';
import Slider from '@material-ui/core/Slider';
import {ClipLoader} from 'react-spinners';
import Box from '@material-ui/core/Box';
import {inpaint, drawImage, changeModel, randomImage} from './util.js';
import './App.css';
// import styles from './App.css';
import PropTypes from 'prop-types';

function Item(props) {
  const { sx, ...other } = props;
  return (
    <Box
      sx={{
        // bgcolor: 'primary.main',
        // color: 'white',
        p: 1,
        m: 1,
        borderRadius: 1,
        // textAlign: 'center',
        fontSize: '1rem',
        fontWeight: '700',
        ...sx,
      }}
      {...other}
    />
  );
}

class Modified extends Component {
    constructor(props) {
        super(props);

        this.state = {
            results: [],
            mouseDown: false,
            clickX: [],
            clickY: [],
            order: false,
            loading: false,
            eraserEnable: false,
            sliderValue: 1
        };
    }

    mouseDown = () => {
       this.setState({
           mouseDown: true,
           clickX: [],
           clickY: []
       }) 
    }

    mouseMove = (evt) => {
        const ctx = this.cPaintDraw.getContext('2d');
        const rect = this.cPaintDraw.getBoundingClientRect();
        const x = evt.clientX - rect.left;
        const y = evt.clientY - rect.top;
        if (this.state.mouseDown) {
            // Drawing from http://www.williammalone.com/articles/create-html5-canvas-javascript-drawing-app/
            const clickX = this.state.clickX;
            const clickY = this.state.clickY;
            clickX.push(x)
            clickY.push(y)

            // 打开橡皮擦
            if (this.state.eraserEnable){
                ctx.save()
                if (clickX.length > 1) {
                    ctx.beginPath();
                    for(let i = 1; i < clickX.length; i++) {		
                        ctx.arc(clickX[i],clickY[i],this.props.brushSize,0,2*Math.PI);
                    }
                    ctx.clip()
                    ctx.clearRect(0,0,256,256);
                }
                ctx.restore();
                
            }else{
                // 251, 150, 107
                ctx.strokeStyle = 'rgba(251, 150, 107, 0.5)';
                ctx.lineJoin = 'round';
                ctx.lineCap = 'round';
                ctx.lineWidth = this.props.brushSize * 2;
    
                if (clickX.length > 1) {
                    ctx.beginPath();
                    ctx.moveTo(clickX[0], clickY[0]);
                    for(let i = 1; i < clickX.length; i++) {		
                        ctx.lineTo(clickX[i], clickY[i]);
                    }
                    ctx.stroke();
                }
            }
           
        } else{
        }
    }

    mouseUp = () => {
        this.setState({
            mouseDown: false,
            loading: true
        }) 

        inpaint(this.cPaintImg.getContext('2d'), this.cPaintDraw.getContext('2d')).then(img => {
            this.cinpaintImg.getContext('2d').putImageData(img[0], 0, 0);
            this.gensliderImage()
         }).then(() => this.setState({ loading: false }));
    }

    mouseLeave = () => {
       this.setState({
           mouseDown: false
       }) 
    }

    gensliderImage(){
        const val = this.state.sliderValue
        if (val>0 && val<256){
            const cPaintImg = this.cPaintImg.getContext('2d').getImageData(0, 0, val, 256);
            this.cGenImg.getContext('2d').putImageData(cPaintImg, 0, 0);
            const cinpaintImg = this.cinpaintImg.getContext('2d').getImageData(val, 0, 256-val+1, 256);
            this.cGenImg.getContext('2d').putImageData(cinpaintImg, val, 0);

            var ctx=this.cGenDraw.getContext("2d");
            ctx.clearRect(0,0,256,256);
            ctx.fillStyle="rgba(251, 150, 107, 0.5)";
            ctx.fillRect(val-1,0,5,256);
        }
        
    }
    handleSlider = (e, val) => {
        this.setState({
            sliderValue: val
        }) 
        this.gensliderImage()
     }

    drawOriginImage = (image) => {
        drawImage(this.cPaintImg.getContext('2d'), image, function(img) {
        }.bind(this));
        drawImage(this.cinpaintImg.getContext('2d'), image, function(img) {
        }.bind(this));
        drawImage(this.cGenImg.getContext('2d'), image, function(img) {
        }.bind(this));
    }
    componentDidMount() {
        this.drawOriginImage(this.props.image);
    }

    componentWillReceiveProps(nProps) {
        if (nProps.reset){
            // 编辑重置
            this.cPaintDraw.getContext('2d').clearRect(0, 0, 256, 256);
        }
        if (this.props.image!=nProps.image) {
            // 图片更换
            this.cPaintDraw.getContext('2d').clearRect(0, 0, 256, 256);
            this.drawOriginImage(nProps.image)
        }
        if (this.props.model!=nProps.model){
            // 模型更换
            changeModel(nProps.model)
        } 
        if (nProps.random){
            // 随机图片
            randomImage(nProps.model).then(img => {
                // 保存随机图片到前端
                nProps.imageRandomed(img[0])
                // 更新编辑图与对应mask
                this.cPaintImg.getContext('2d').putImageData(img[0], 0, 0);
                this.cPaintDraw.getContext('2d').putImageData(img[1], 0, 0);
                // 修复好的图片
                this.cinpaintImg.getContext('2d').putImageData(img[2], 0, 0);
                // 根据滑动条更新合成图
                this.gensliderImage()
             }).then(() => this.setState({ loading: false }));
        }
        if (this.props.eraserEnable!=nProps.eraserEnable){
            this.setState({
                eraserEnable: nProps.eraserEnable
            }) 
        }
        this.props = nProps;
    }

    render() {
        return (
            <div>
                <Box
                    sx={{
                    textAlign: 'center',
                    display: 'flex',
                    justifyContent: 'space-around',
                    flexDirection: 'row',
                    p: 1,
                    m: 1,
                    }}
                >
                    <Item>
                        <Paper style={{height: 256, width: 256, display: "inline-block"}} zDepth={3}>
                            <canvas id="modified-canvas" height="256px" width="256px" 
                                    ref={cPaintImg => this.cPaintImg = cPaintImg}> 
                            </canvas>
                            <canvas id="draw-canvas" height="256px" width="256px" 
                                    ref={cPaintDraw => this.cPaintDraw = cPaintDraw} onMouseDown={this.mouseDown}
                                    onMouseMove={this.mouseMove} onMouseUp={this.mouseUp}
                                    onMouseLeave={this.mouseLeave}>
                            </canvas>
                        </Paper>
                        <h3 style={{margin:10}} id="modified-title">编辑图片</h3>
                        <ClipLoader color="rgb(63, 81, 181)" loading={this.state.loading} />
                    </Item>
                    <Item >
                        <Paper style={{height: 256, width: 256}} zDepth={3}>
                        <canvas id="original-canvas" height="256px" width="256px" ref={c => this.cGenImg = c}></canvas>
                        <canvas id="draw-canvas" height="256px" width="256px" ref={c => this.cGenDraw = c}>
                        </canvas>
                        </Paper>  
                        <h3 style={{margin:10}} id="modified-title">智能修复</h3>
                        <Slider
                            min={1}
                            max={256}
                            step={1}
                            value={this.state.sliderValue}
                            onChange={this.handleSlider}
                            size="large"
                            className='slider' 
                            componentsProps={{ thumb: { className: 'thumb' } }}
                        />
                        <view style={{width:0,height:0,overflow:"hidden",display:"None"}}>
                        <canvas id="original-canvas" height="256px" width="256px" ref={c => this.cinpaintImg = c}></canvas>
                        </view>
                    </Item>
                </Box>
            </div>
        );
    }
}

export default Modified;
