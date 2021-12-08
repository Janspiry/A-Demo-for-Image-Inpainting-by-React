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
import ActionDelete from 'material-ui/svg-icons/action/delete';
import ImageAddAPhoto from 'material-ui/svg-icons/image/add-a-photo';
import ActionAutorenew from 'material-ui/svg-icons/action/autorenew';
import ActionRestore from 'material-ui/svg-icons/action/restore';
import FileFileDownload from 'material-ui/svg-icons/file/file-download';
import PropTypes from 'prop-types';
import './App.css';

function Item(props) {
  const { sx, ...other } = props;
  return (
    <Box
      sx={{
        p: 1,
        m: 0,
        borderRadius: 1,
        // fontSize: '1rem',
        // fontWeight: '700',
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
      
      <div className="tool-bar-wrap">
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'row',
            // bgcolor: 'white',
            justifyContent: 'space-between',
            alignItems: 'center', 
            p: 0,
            m: 0,
            paddingLeft: 4,
            paddingRight: 4
            // bgcolor: 'background.paper',
          }}
        >
          <Item sx={{textAlign: 'center'}}>
              <FormControl variant="standard" fullWidth size>
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
          <Divider orientation="vertical" flexItem />  
          <Item >
              <Box sx={{  display: 'flex', flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',}}>
                画笔尺寸
                <Box sx={{ marginLeft:2, marginRight:3, minWidth: 120}} >
                <Slider min={2} max={30} defaultValue={15} step={1}  onChange={this.props.brushChanged}/>
                </Box>
                <svg height="60px" width="60px">
                  <circle cx="30" cy="30" fill={indigo500} r={this.props.brushSize}/>
                </svg>
              </Box>
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
          <Divider orientation="vertical" flexItem />
          <Item>
              <Button variant="text" color="error" startIcon={<ActionRestore  color="#d32f2f"/>} onClick={this.props.reset}>重修一次</Button>
          </Item>
          <Divider orientation="vertical" flexItem />
          <Item>
              <Button startIcon={<ActionAutorenew  color="#1976d2"/>} variant="text" onClick={this.props.randomImage}>随机图片</Button>
              
          </Item>
          
          <Item>
              <Stack direction="row" alignItems="center" spacing={2}>
                <label htmlFor="contained-button-file">
                  <Input accept="image/*" id="contained-button-file" multiple type="file" className="file" ref={x=>this._file=x} onChange={this.fileChange}/>
                  <Button  startIcon={<ImageAddAPhoto  color="#1976d2"/>} variant="text" component="span" onClick={this.props.imageUpload}>
                    上传图片
                  </Button>
                </label>
              </Stack>    
          </Item>
  
          <Item>
              <Button    startIcon={<FileFileDownload  color="#1976d2"/>} variant="text" onClick={this.props.download}>保存图片</Button>

          </Item>
               
        </Box>
          

        
        
      </div>
      
    );
    
        
    
  }
}

export default Options;
