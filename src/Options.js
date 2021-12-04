import React, { Component } from 'react';
import Button from '@material-ui/core/Button';
import Box from '@material-ui/core/Box';
import Slider from '@material-ui/core/Slider';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import FormControl from '@material-ui/core/FormControl';
import Select from '@material-ui/core/Select';
import FormGroup from '@material-ui/core/FormGroup';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Switch from '@material-ui/core/Switch';
import Divider from '@material-ui/core/Divider';

import Stack from '@material-ui/core/Stack';
import {indigo500} from 'material-ui/styles/colors';
import { styled } from '@material-ui/core/styles';
import LoadingButton from '@material-ui/core/Button';

import PropTypes from 'prop-types';
import './App.css';

function Item(props) {
  const { sx, ...other } = props;
  return (
    <Box
      sx={{
        p: 1,
        m: 1,
        borderRadius: 1,
        fontSize: '1rem',
        fontWeight: '700',
        ...sx,
      }}
      {...other}
    />
  );
}
const Input = styled('input')({
  display: 'none',
});
class Options extends Component {
  constructor(props) {
    super(props);
    
    this.state = {
      
    };
  }
  
  handleImageUpload(e) {
    this.props.imageUpload(e)
  }

  componentWillReceiveProps(nProps) {
    this.props = nProps
  }

  render() {
    return (
      
      <div className="box" id="options">
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            p: 1,
            m: 1,
            // bgcolor: 'background.paper',
          }}
        >
          <Item sx={{ minWidth: 200 , textAlign: 'center'}}>
              <FormControl fullWidth>
                <InputLabel id="demo-simple-select-label">选择图片类型</InputLabel>
                <Select
                  autoWidth
                  onChange={this.props.imageTypeChanged} value={this.props.image_type} 
                >
                  <MenuItem value="celebahq">处理人脸</MenuItem>
                  <MenuItem value="places2">处理自然风景</MenuItem>
                </Select>
              </FormControl>
          </Item>
          <Item sx={{ minWidth: 200, textAlign: 'center' }}>
          
              <FormControl fullWidth>
                <InputLabel id="demo-simple-select-label">选择模型</InputLabel>
                <Select onChange={this.props.maskModeChanged} value={this.props.mask_mode} >
                  <MenuItem value="gconv">应付随机缺失</MenuItem>
                  <MenuItem value="center">应付大面积缺失</MenuItem>
                </Select>
              </FormControl>

          </Item>
          <Divider/>
          <Item>
              <h4>画笔尺寸</h4>    
              <Box sx={{  display: 'flex', flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',}}>
                <Box sx={{ minWidth: 120}} >
                <Slider min={2} max={30} defaultValue={15} step={1}  onChange={this.props.brushChanged}/>
                </Box>
                <svg height="60px" width="60px">
                  <circle cx="30" cy="30" fill={indigo500} r={this.props.brushSize}/>
                </svg>
              </Box>
          </Item>
          <Divider/>
          <Item>
              <Button variant="outlined" color="error" onClick={this.props.reset}>重修一次</Button>
          </Item>
          <Item >
              <FormGroup>
                  <FormControlLabel
                    control={
                      <Switch checked={this.props.eraserEnable} onChange={this.props.eraserChanged} name="gilad" />
                    }
                    labelPosition="right"
                    label="使用橡皮擦"
                  />
              </FormGroup>
          </Item>
          <Divider/>
          <Item  sx={{  display: 'flex', flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between'}}>
              <Button variant="outlined" onClick={this.props.randomImage}>随机图片</Button>
              <Stack direction="row" alignItems="center" spacing={2}>
                <label htmlFor="contained-button-file">
                  <Input accept="image/*" id="contained-button-file" multiple type="file" onChange={this.handleImageUpload.bind(this)}/>
                  <Button variant="contained" component="span">
                    上传图片
                  </Button>
                </label>
              </Stack>
          </Item>
          
        </Box>
          

        
        
      </div>
      
    );
    
        
    
  }
}

export default Options;
