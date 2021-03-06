import React, { Component } from 'react';

import Paper  from '@material-ui/core/Paper';
import Slider from '@material-ui/core/Slider';
import {ClipLoader} from 'react-spinners';
import Stack from '@material-ui/core/Stack';
import Snackbar from '@material-ui/core/Snackbar';
import MuiAlert from '@material-ui/core/Alert';

import Box from '@material-ui/core/Box';
import {inpaint, changeModel, randomImage} from './util.js';
import './App.css';
// import styles from './App.css';
import PropTypes from 'prop-types';
import FileSaver from 'file-saver';
const Alert = React.forwardRef(function Alert(props, ref) {
return <MuiAlert elevation={6} ref={ref} variant="filled" {...props} />;
});

function Item(props) {
  const { sx, ...other } = props;
  return (
    <Box
      sx={{
        // bgcolor: 'primary.main',
        color: 'black',
        p: 1,
        m: 1,
        borderRadius: 10,
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
            sliderValue: 1,
            error: false,
            errorMsg: '加载错误，请刷新后重试',
            showElem:false
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
                    ctx.clearRect(0,0,512,512);
                }
                ctx.restore();
                
            }else{
                // 200, 224, 228
                ctx.strokeStyle = 'rgba(200, 224, 228, 0.8)';
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
        this.inpaintImage()
    }

    mouseLeave = () => {
       this.setState({
           mouseDown: false
       }) 
    }

    // 根据滑动条生成图片
    gensliderImage(){
        const val = this.state.sliderValue
        if (val>0 && val<512){
            const cPaintImg = this.cPaintImg.getContext('2d').getImageData(0, 0, val, 512);
            this.cGenImg.getContext('2d').putImageData(cPaintImg, 0, 0);
            const cinpaintImg = this.cinpaintImg.getContext('2d').getImageData(val, 0, 512-val+1, 512);
            this.cGenImg.getContext('2d').putImageData(cinpaintImg, val, 0);

            var ctx=this.cGenDraw.getContext("2d");
            ctx.clearRect(0,0,512,512);
            ctx.fillStyle="rgba(200, 224, 228, 1)";
            ctx.fillRect(val-1,0,5,512);
        }
        
    }
    
    // 调节滑动条
    handleSlider = (e, val) => {
        this.setState({
            sliderValue: val
        }) 
        this.gensliderImage()
     }
    
    // 调用后端修复图片
    inpaintImage = () => {
        this.setState({
            mouseDown: false,
            loading: true
        })
        inpaint(this.cPaintImg.getContext('2d'), this.cPaintDraw.getContext('2d')).then(img => {
            if (img==null){
                this.handleError('模型调用失败，请稍后再试')
            }else{
                this.cinpaintImg.getContext('2d').putImageData(img[0], 0, 0);
                this.gensliderImage()
            }
            
         }).then(() => this.setState({ loading: false }));
    }

    // 从后端随机一张图并修复
    drawRandomImage = (nProps) => {
        this.setState({
            mouseDown: false,
            loading: true
        })
        randomImage(nProps.mask_mode, nProps.mask_mode).then(img => {
            if (img==null){
                this.handleError('随机图片失败，请稍后再试')
            }else{
                // 更新编辑图与对应mask
                this.cPaintImg.getContext('2d').putImageData(img[0], 0, 0);
                this.cPaintDraw.getContext('2d').putImageData(img[1], 0, 0);
                // 修复好的图片
                this.cinpaintImg.getContext('2d').putImageData(img[2], 0, 0);
                // 根据滑动条更新合成图
                this.gensliderImage()
            }
            
         }).then(() => this.setState({ loading: false }));
    }
    componentDidMount() {
        this.drawRandomImage(this.props);
    }

    componentWillReceiveProps(nProps) {
        if (nProps.reset){
            // 编辑重置
            this.cPaintDraw.getContext('2d').clearRect(0, 0, 512, 512);
        }
        if (this.props.image_type!=nProps.image_type || this.props.mask_mode!=nProps.mask_mode) {
            // 图片类型或者模型更换
            this.setState({
                loading: true
            })
            changeModel(nProps.image_type, nProps.mask_mode).then(result =>{
                if (result){
                    this.inpaintImage()
                }else{
                    this.handleError('模型更换失败，请稍后重试')
                }
            }).then(() => this.setState({ loading: false }));
        }
        if (nProps.random){
            // 随机图片
            this.drawRandomImage(this.props);
        }
        if (this.props.eraserEnable!=nProps.eraserEnable){
            // 橡皮擦
            this.setState({
                eraserEnable: nProps.eraserEnable
            }) 
        }
        if(nProps.download){
            this.adownLoad(nProps)
        } 
        this.props = nProps;
    }

    // 加载错误，提示信息
    handleError = (msg) => {
        // console.log(msg)
        this.setState({
            error: true,
            errorMsg: msg
        })
    };

    // 关闭错误提示
    handleErrorClose = (event, reason) => {
        if (reason === 'clickaway') {
        return;
        }
        this.setState({
            error: false
        })
    };
    render() {
        const { showElem }= false;
        return (
            
            <div className="drawing-box">
                <Stack spacing={2} sx={{ width: '100%' }}>
                <Snackbar open={this.state.error}  autoHideDuration={3000} onClose={this.handleErrorClose}>
                    <Alert onClose={this.handleErrorClose} severity="error" sx={{ width: '100%' }} >
                        {this.state.errorMsg}
                    </Alert>
                </Snackbar>
                </Stack>
                <Box
                    sx={{
                    textAlign: 'center',
                    display: 'flex',
                    flexDirection: 'row',
                    justifyContent: 'space-around',
                    alignContent: 'center',
                    
                    bottom: 0,
                    p: 1,
                    m: 1,
                    paddingTop: 10
                    }}
                >
                    <Item>
                        <Stack direction="row" alignItems="center" spacing={2}>
                            <label htmlFor="contained-button-file2">
                            </label>
                        </Stack>
                    </Item>
                    <Item >
                        <h3 style={{margin:10}} id="modified-title">编辑图片</h3>

                        <Paper elevation={24} style={{height: 512, width: 512}}>
                            <canvas id="modified-canvas" height="512px" width="512px" 
                                    ref={cPaintImg => this.cPaintImg = cPaintImg}> 
                            </canvas>
                            <canvas id="draw-canvas" height="512px" width="512px" 
                                    ref={cPaintDraw => this.cPaintDraw = cPaintDraw} onMouseDown={this.mouseDown}
                                    onMouseMove={this.mouseMove} onMouseUp={this.mouseUp}
                                    onMouseLeave={this.mouseLeave}>
                            </canvas>
                        </Paper>
                        <ClipLoader color="rgb(25, 118, 210)" loading={this.state.loading} />
                        {/* <LinearProgress  disabled={this.state.loading}/> */}
                    </Item>
                    <Item >
                    <h3 style={{margin:10}} id="modified-title">智能修复</h3>

                        <Paper elevation={24} style={{height: 512, width: 512}}>
                        <canvas id="original-canvas" height="512px" width="512px" ref={c => this.cGenImg = c}></canvas>
                        <canvas id="draw-canvas" height="512px" width="512px" ref={c => this.cGenDraw = c}>
                        </canvas>
                        </Paper>  
                        <Slider
                            min={1}
                            max={512}
                            step={1}
                            value={this.state.sliderValue}
                            onChange={this.handleSlider}
                            className='slider' 
                            componentsProps={{ thumb: { className: 'thumb' } }}
                        />
                        <div style={{width:0,height:0,overflow:"hidden",display:"None"}}>
                        <canvas id="original-canvas" height="512px" width="512px" ref={c => this.cinpaintImg = c}></canvas>
                        </div>
                    </Item>
                    <Item>
                        <Stack direction="row" alignItems="center" spacing={2}>
                            <label htmlFor="contained-button-file">
                                <input accept="image/*" id="contained-button-file" multiple type="file" className="file" ref={x=>this._file=x} onChange={this.fileChange} style={{display:showElem?'block':'none'}}/>
                            

                            </label>
                        </Stack>
                    </Item>
                </Box>
            </div>
        );
    }
    fileChange=(nProps)=>{
        this.setState({S:true});
        console.log(nProps.imageUpload);
        // console.log(nProps.reset);
        let picom= this._file.files[0];
        if(!picom) return ;
        // console.log(picom);
        let fileReade=new FileReader();
        fileReade.readAsDataURL(picom);
        fileReade.onload=ev => {
            this.img=new Image();
            this.img.src=ev.target.result;
            this.img.onload=()=>{
                this.ctx=this.cPaintImg.getContext('2d');
                this.ctx.drawImage(this.img,0,0,512,512);
                this.ctx=this.cGenDraw.getContext('2d');
                this.ctx.drawImage(this.img,0,0,512,512);
                // this.ctx=this.cGenDraw.getContext('2d');
                this.inpaintImage()
            };
        }
    }
    adownLoad=(nProps)=>{
        // this.setState({S:true});
        console.log(nProps.download);
        var canvas = document.getElementById("original-canvas");
        canvas.toBlob(function(blob) {
            FileSaver.saveAs(blob, "pretty image.png");
        });
    }
}

export default Modified;
