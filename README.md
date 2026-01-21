## Stan 文档中文本地化项目

### 项目简介

本代码库存放了 [Stan 文档](https://mc-stan.org/docs/) 中文版本的源文件。当前我们正在翻译[《Stan 用户指南》](https://mc-stan.org/docs/stan-users-guide/)，您可以通过以下命令构建翻译后的手册：

```bash
cd stan-users-guide
quarto render --to html
```

您也可以通过[此链接](https://stan-cn.netlify.app/)预览已渲染的文档内容。

项目中的 `upstream` 目录存放了原始的 Stan 文档源文件，主要用于追溯[上游仓库](https://github.com/stan-dev/docs/tree/master/src)的更新内容。

### 致谢

本文档由[上海财经大学贝叶斯兴趣小组（SUFE-Bayes）](https://github.com/sufe-bayes)翻译完成。

本项目得到了 [NumFOCUS Small Development Grant](https://numfocus.org/programs/small-development-grants) 的支持。



## Chinese Translation of Stan Documentation

### Introduction

This repository contains the source files of the Chinese version of
[Stan documentation](https://mc-stan.org/docs/).
Right now we are working on the
[Stan User's Guide](https://mc-stan.org/docs/stan-users-guide/),
and you can build the translated book using the following command:

```bash
cd stan-users-guide
quarto render --to html
```

Alternatively, you can preview the rendered output
[here](https://stan-cn.netlify.app/).

The `upstream` directory contains the source files of the original
Stan documentation. We mainly use this directory to track the changes in
[the upstream repository](https://github.com/stan-dev/docs/tree/master/src).

### Acknowledgement

This project is maintained by the [SUFE-Bayes](https://github.com/sufe-bayes) special interest group.

This project is supported by the
[NumFOCUS Small Development Grant](https://numfocus.org/programs/small-development-grants).
