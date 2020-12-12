# Django-MNIST 项目

## 目标

通过此项目，熟悉 Django web 框架常用到的知识点，了解 PyTorch 框架，熟悉平时项目中会用到的工具，
包括 git、pycharm、XShell、POSTMAN 等工具软件，了解 web 应用工作原理

你需要通过 Django 框架实现一个 web 版的手写体识别项目，主要功能为：

1. web 前端基础功能是上传数字图片，扩展功能是利用前端画板完成数字的手写绘制（提示：canvas或开源库）
2. 后端调用 CNN 模型识别后将结果展示在 web 前端上
3. 将完成的 web 应用部署到 Linux 服务器，要求：
    - Nginx 作为反向代理服务器，处理静态资源
    - WSGI 服务器使用 gunicorn/uwsgi （二选一）
    - supervisore 管理应用进程
4. 在完成上一步后，可完成应用的 docker 部署，以简化项目部署过程
5. 初学阶段，前端的重点在于与后端的交互，不必拘泥于样式，如配色、背景等，但页面需要简洁直观，要做到根据屏幕尺寸自适应。

## 可能用到的技术点

当前代码已经实现了 Pytorch MNIST CNN 模型的手写体识别功能，且已包含训练好的模型参数，你可以直接调用其中的 API 
完成手写体识别功能，你也可以按需修改其中的代码。你主要需要做的是完成 web 应用的逻辑实现部分代码，在实现了 Django 后端 API 代码后，
可通过 POSTMAN 工具测试其是否能正常工作，然后完成前端部分代码，前端部分建议先写原生js代码以达到锻炼的目的，在初步完成的基础上可使用 JQuery、Vue.js 等框架进行完善。

以下为可能对你有帮助的资料:

1. 《Django企业开发实战 高效Python Web框架指南》
2. 前端学习路线: 
    - HTML、CSS基础知识：https://www.w3cschool.cn/tutorial
    - JS基础知识：https://www.runoob.com/js/js-tutorial.html
    - Vue.js：https://cn.vuejs.org/v2/guide/
    - JS资源大全：https://github.com/jobbole/awesome-javascript-cn
    - 大前端综合教程、资源汇总：https://github.com/nicejade/nice-front-end-tutorial/blob/master/tutorial/front-end-tutorial.md
3. POSTMAN: https://blog.csdn.net/flowerspring/article/details/52774399
4. Chrome 调试工具: 
    - https://www.cnblogs.com/laixiangran/p/8777579.html
    - https://developers.google.com/web/tools/chrome-devtools?hl=zh-cn
5. pycharm 及 Anaconda: 
    - https://blog.csdn.net/makingLJ/article/details/78929055
    - https://blog.csdn.net/makingLJ/article/details/98109652
6. XShell: https://blog.csdn.net/makingLJ/article/details/86355938
7. Git 相关:
    - git 教程: https://git-scm.com/book/zh/v2
    - SourceTree: https://www.cnblogs.com/Can-daydayup/p/13128633.html
8. Linux 常见工具相关:
    - Linux : https://linuxtools-rst.readthedocs.io/zh_CN/latest/index.html
9. Docker 相关:
    - https://yeasy.gitbook.io/docker_practice/

有遇到不会的，或者在实现过程中有遇到 bug，请善用 google 搜索引擎解决

## 时间

两到三周完成，每人一份

## 作业上传方式

可以前后端分工完成，也可以每个人完成一份，但鼓励单人完成，以帮助更好地理解整个框架的工作方式。
完成后，请在 gitlab 上单独创建一个 project，上传代码
