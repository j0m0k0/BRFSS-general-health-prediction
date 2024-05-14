import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, TablePagination } from '@material-ui/core';

const Dataset = () => {
  const [data, setData] = useState([]);
  const [page, setPage] = useState(1);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [columns, setColumns] = useState([]);

  const fetchData = async (page, pageSize) => {
    const res = await axios.get(`http://localhost:5000/dataset/items?page=${page}&pagesize=${pageSize}`);
    setData(res.data);
    if (res.data.length > 0) {
      setColumns(Object.keys(res.data[0]).slice(0, 10));
    }
  };

  useEffect(() => {
    fetchData(page, rowsPerPage);
  }, [page, rowsPerPage]);

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = event => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(1);
  };

  const renderCell = (item, columnId) => {
    const content = item[columnId];

    if (content && typeof content === 'object' && content.hasOwnProperty('$oid')) {
      return content['$oid'];
    }

    return content;
  }

  return (
    <div>
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              {columns.map((columnId) => (
                <TableCell key={columnId}>{columnId}</TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {data.map((item, index) => (
              <TableRow key={index}>
                {columns.map((columnId) => (
                  <TableCell key={columnId}>{renderCell(item, columnId)}</TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
        <TablePagination
          component="div"
          count={-1} // the api should provide total count of data for full pagination
          page={page}
          onChangePage={handleChangePage}
          rowsPerPage={rowsPerPage}
          onChangeRowsPerPage={handleChangeRowsPerPage}
        />
      </TableContainer>
    </div>
  );
};

export default Dataset;
